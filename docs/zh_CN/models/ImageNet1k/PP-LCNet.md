# PP-LCNet 系列
---

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型细节](#1.2)
      - [1.2.1 更好的激活函数](#1.2.1)
      - [1.2.2 合适的位置添加 SE 模块](#1.2.2)
      - [1.2.3 合适的位置添加更大的卷积核](#1.2.3)
      - [1.2.4 GAP 后使用更大的 1x1 卷积层](#1.2.4)
    - [1.3 实验结果](#1.3)
    - [1.4 Benchmark](#1.4)
      - [1.4.1 基于 Intel Xeon Gold 6148 的预测速度](#1.4.1)
      - [1.4.2 基于 V100 GPU 的预测速度](#1.4.2)
      - [1.4.3 基于 SD855 的预测速度](#1.4.3)
- [2. 模型快速体验](#2)
    - [2.1 安装 paddlepaddle](#2.1)
    - [2.2 安装 paddleclas](#2.2)
    - [2.3 预测](#2.3)
- [3. 模型训练、评估和预测](#3)
    - [3.1 环境配置](#3.1)
    - [3.2 数据准备](#3.2)
    - [3.3 模型训练](#3.3)
      - [3.3.1 训练 ImageNet](#3.3.1)
      - [3.3.2 基于 ImageNet 权重微调](#3.3.2)
    - [3.4 模型评估](#3.4)
    - [3.5 模型预测](#3.5)
- [4. 模型推理部署](#4)
  - [4.1 推理模型准备](#4.1)
    - [4.1.1 基于训练得到的权重导出 inference 模型](#4.1.1)
    - [4.1.2 直接下载 inference 模型](#4.1.2)
  - [4.2 基于 Python 预测引擎推理](#4.2)
    - [4.2.1 预测单张图像](#4.2.1)
    - [4.2.2 基于文件夹的批量预测](#4.2.2)
  - [4.3 基于 C++ 预测引擎推理](#4.3)
  - [4.4 服务化部署](#4.4)
  - [4.5 端侧部署](#4.5)
  - [4.6 Paddle2ONNX 模型转换与预测](#4.6)
- [5. 引用](#5)

<a name="1"></a>

## 1. 模型介绍

### 1.1 模型简介

在计算机视觉领域中，骨干网络的好坏直接影响到整个视觉任务的结果。在之前的一些工作中，相关的研究者普遍将 FLOPs 或者 Params 作为优化目的，但是在工业界真实落地的场景中，推理速度才是考量模型好坏的重要指标，然而，推理速度和准确性很难兼得。考虑到工业界有很多基于 Intel CPU 的应用，所以我们本次的工作旨在使骨干网络更好的适应 Intel CPU，从而得到一个速度更快、准确率更高的轻量级骨干网络，与此同时，目标检测、语义分割等下游视觉任务的性能也同样得到提升。

近年来，有很多轻量级的骨干网络问世，尤其最近两年，各种 NAS 搜索出的网络层出不穷，这些网络要么主打 FLOPs 或者 Params 上的优势，要么主打 ARM 设备上的推理速度的优势，很少有网络专门针对 Intel CPU 做特定的优化，导致这些网络在 Intel CPU 端的推理速度并不是很完美。基于此，我们针对 Intel CPU 设备以及其加速库 MKLDNN 设计了特定的骨干网络 PP-LCNet，比起其他的轻量级的 SOTA 模型，该骨干网络可以在不增加推理时间的情况下，进一步提升模型的性能，最终大幅度超越现有的 SOTA 模型。与其他模型的对比图如下。
![](../../../images/PP-LCNet/PP-LCNet-Acc.png)

<a name="1.2"></a>

### 1.2 模型细节

网络结构整体如下图所示。
![](../../../images/PP-LCNet/PP-LCNet.png)
我们经过大量的实验发现，在基于 Intel CPU 设备上，尤其当启用 MKLDNN 加速库后，很多看似不太耗时的操作反而会增加延时，比如 elementwise-add 操作、split-concat 结构等。所以最终我们选用了结构尽可能精简、速度尽可能快的 block 组成我们的 BaseNet（类似 MobileNetV1）。基于 BaseNet，我们通过实验，总结了四条几乎不增加延时但是可以提升模型精度的方法，融合这四条策略，我们组合成了 PP-LCNet。下面对这四条策略一一介绍：

<a name="1.2.1"></a>

#### 1.2.1 更好的激活函数

自从卷积神经网络使用了 ReLU 激活函数后，网络性能得到了大幅度提升，近些年 ReLU 激活函数的变体也相继出现，如 Leaky-ReLU、P-ReLU、ELU 等，2017 年，谷歌大脑团队通过搜索的方式得到了 swish 激活函数，该激活函数在轻量级网络上表现优异，在 2019 年，MobileNetV3 的作者将该激活函数进一步优化为 H-Swish，该激活函数去除了指数运算，速度更快，网络精度几乎不受影响。我们也经过很多实验发现该激活函数在轻量级网络上有优异的表现。所以在 PP-LCNet 中，我们选用了该激活函数。

<a name="1.2.2"></a>

#### 1.2.2 合适的位置添加 SE 模块

SE 模块是 SENet 提出的一种通道注意力机制，可以有效提升模型的精度。但是在 Intel CPU 端，该模块同样会带来较大的延时，如何平衡精度和速度是我们要解决的一个问题。虽然在 MobileNetV3 等基于 NAS 搜索的网络中对 SE 模块的位置进行了搜索，但是并没有得出一般的结论，我们通过实验发现，SE 模块越靠近网络的尾部对模型精度的提升越大。下表也展示了我们的一些实验结果：

| SE Location       | Top-1 Acc(\%) | Latency(ms) |
|:--:|:--:|:--:|
| 1100000000000     | 61.73           | 2.06         |
| 0000001100000     | 62.17           | 2.03         |
| <b>0000000000011<b>     | <b>63.14<b>           | <b>2.05<b>         |
| 1111111111111     | 64.27           | 3.80         |

最终，PP-LCNet 中的 SE 模块的位置选用了表格中第三行的方案。

<a name="1.2.3"></a>

#### 1.2.3 合适的位置添加更大的卷积核

在 MixNet 的论文中，作者分析了卷积核大小对模型性能的影响，结论是在一定范围内大的卷积核可以提升模型的性能，但是超过这个范围会有损模型的性能，所以作者组合了一种 split-concat 范式的 MixConv，这种组合虽然可以提升模型的性能，但是不利于推理。我们通过实验总结了一些更大的卷积核在不同位置的作用，类似 SE 模块的位置，更大的卷积核在网络的中后部作用更明显，下表展示了 5x5 卷积核的位置对精度的影响：

| large-kernel Location       | Top-1 Acc(\%) | Latency(ms) |
|:--:|:--:|:--:|
| 1111111111111     | 63.22           | 2.08         |
| 1111111000000     | 62.70           | 2.07        |
| <b>0000001111111<b>     | <b>63.14<b>           | <b>2.05<b>         |

实验表明，更大的卷积核放在网络的中后部即可达到放在所有位置的精度，与此同时，获得更快的推理速度。PP-LCNet 最终选用了表格中第三行的方案。

<a name="1.2.4"></a>

#### 1.2.4 GAP 后使用更大的 1x1 卷积层

在 GoogLeNet 之后，GAP（Global-Average-Pooling）后往往直接接分类层，但是在轻量级网络中，这样会导致 GAP 后提取的特征没有得到进一步的融合和加工。如果在此后使用一个更大的 1x1 卷积层（等同于 FC 层），GAP 后的特征便不会直接经过分类层，而是先进行了融合，并将融合的特征进行分类。这样可以在不影响模型推理速度的同时大大提升准确率。
BaseNet 经过以上四个方面的改进，得到了 PP-LCNet。下表进一步说明了每个方案对结果的影响：

| Activation | SE-block | Large-kernel | last-1x1-conv | Top-1 Acc(\%) | Latency(ms) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 0       | 1       | 1               | 1                | 61.93 | 1.94 |
| 1       | 0       | 1               | 1                | 62.51 | 1.87 |
| 1       | 1       | 0               | 1                | 62.44 | 2.01 |
| 1       | 1       | 1               | 0                | 59.91 | 1.85 |
| <b>1<b>       | <b>1<b>       | <b>1<b>               | <b>1<b>                | <b>63.14<b> | <b>2.05<b> |

<a name="1.3"></a>

### 1.3 实验结果

<a name="1.3.1"></a>

#### 1.3.1 图像分类

图像分类我们选用了 ImageNet 数据集，相比目前主流的轻量级网络，PP-LCNet 在相同精度下可以获得更快的推理速度。当使用百度自研的 SSLD 蒸馏策略后，精度进一步提升，在 Intel cpu 端约 5ms 的推理速度下 ImageNet 的 Top-1 Acc 超过了 80%。

| Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) | 预训练模型下载地址 | inference模型下载地址 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PPLCNet_x0_25  | 1.5 | 18  | 51.86 | 75.65 | 0.44 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_25_infer.tar) |
| PPLCNet_x0_35  | 1.6 | 29  | 58.09 | 80.83 | 0.45 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_35_infer.tar) |
| PPLCNet_x0_5   | 1.9 | 47  | 63.14 | 84.66 | 0.44 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_5_infer.tar) |
| PPLCNet_x0_75  | 2.4 | 99  | 68.18 | 88.30 | 0.44 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_75_infer.tar) |
| PPLCNet_x1_0     | 3.0 | 161 | 71.32 | 90.03 | 0.47 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar) |
| PPLCNet_x1_5   | 4.5 | 342 | 73.71 | 91.53 | 0.54 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_5_infer.tar) |
| PPLCNet_x2_0     | 6.5 | 590 | 75.18 | 92.27 | 0.64 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x2_0_infer.tar) |
| PPLCNet_x2_5   | 9.0 | 906 | 76.60 | 93.00 | 0.71 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x2_5_infer.tar) |
| PPLCNet_x0_5_ssld | 1.9 | 47  | 66.10 | 86.46 | 0.44 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x0_5_ssld_infer.tar) |
| PPLCNet_x1_0_ssld | 3.0 | 161 | 74.39 | 92.09 | 0.47 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_ssld_infer.tar) |
| PPLCNet_x2_5_ssld | 9.0 | 906 | 80.82 | 95.33 | 0.71 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_ssld_pretrained.pdparams) | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x2_5_ssld_infer.tar) |

其中 `_ssld` 表示使用 `SSLD 蒸馏`后的模型。关于 `SSLD蒸馏` 的内容，详情 [SSLD 蒸馏](../../training/advanced/knowledge_distillation.md)。

与其他轻量级网络的性能对比：

| Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileNetV2_x0_25  | 1.5 | 34  | 53.21 | 76.52 | 2.47 |
| MobileNetV3_small_x0_35  | 1.7 | 15  | 53.03 | 76.37 | 3.02 |
| ShuffleNetV2_x0_33  | 0.6 | 24  | 53.73 | 77.05 | 4.30 |
| <b>PPLCNet_x0_25<b>  | <b>1.5<b> | <b>18<b>  | <b>51.86<b> | <b>75.65<b> | <b>1.74<b> |
| MobileNetV2_x0_5  | 2.0 | 99  | 65.03 | 85.72 | 2.85 |
| MobileNetV3_large_x0_35  | 2.1 | 41  | 64.32 | 85.46 | 3.68 |
| ShuffleNetV2_x0_5  | 1.4 | 43  | 60.32 | 82.26 | 4.65 |
| <b>PPLCNet_x0_5<b>   | <b>1.9<b> | <b>47<b>  | <b>63.14<b> | <b>84.66<b> | <b>2.05<b> |
| MobileNetV1_x1_0 | 4.3 | 578  | 70.99 | 89.68 | 3.38 |
| MobileNetV2_x1_0 | 3.5 | 327  | 72.15 | 90.65 | 4.26 |
| MobileNetV3_small_x1_25  | 3.6 | 100  | 70.67 | 89.51 | 3.95 |
| <b>PPLCNet_x1_0<b>     |<b> 3.0<b> | <b>161<b> | <b>71.32<b> | <b>90.03<b> | <b>2.46<b> |

<a name="1.3.2"></a>

#### 1.3.2 目标检测

目标检测的方法我们选用了百度自研的 PicoDet，该方法主打轻量级目标检测场景，下表展示了在 COCO 数据集上、backbone 选用 PP-LCNet 与 MobileNetV3 的结果的比较，无论在精度还是速度上，PP-LCNet 的优势都非常明显。

| Backbone | mAP(%) | Latency(ms) |
|:--:|:--:|:--:|
MobileNetV3_large_x0_35 | 19.2 | 8.1 |
<b>PPLCNet_x0_5<b> | <b>20.3<b> | <b>6.0<b> |
MobileNetV3_large_x0_75 | 25.8 | 11.1 |
<b>PPLCNet_x1_0<b> | <b>26.9<b> | <b>7.9<b> |

<a name="1.3.3"></a>

#### 1.3.3 语义分割

语义分割的方法我们选用了 DeeplabV3+，下表展示了在 Cityscapes 数据集上、backbone 选用 PP-LCNet 与 MobileNetV3 的比较，在精度和速度方面，PP-LCNet 的优势同样明显。

| Backbone | mIoU(%) | Latency(ms) |
|:--:|:--:|:--:|
MobileNetV3_large_x0_5 | 55.42 | 135 |
<b>PPLCNet_x0_5<b> | <b>58.36<b> | <b>82<b> |
MobileNetV3_large_x0_75 | 64.53 | 151 |
<b>PPLCNet_x1_0<b> | <b>66.03<b> | <b>96<b> |

<a name="1.4"></a>

## 1.4 Benchmark

<a name="1.4.1"></a>

#### 1.4.1 基于 Intel Xeon Gold 6148 的预测速度  

| Model | Latency(ms)<br/>bs=1, thread=10 |
|:--:|:--:|
| PPLCNet_x0_25  | 1.74 |
| PPLCNet_x0_35  | 1.92 |
| PPLCNet_x0_5   | 2.05 |
| PPLCNet_x0_75  | 2.29 |
| PPLCNet_x1_0   | 2.46 |
| PPLCNet_x1_5   | 3.19 |
| PPLCNet_x2_0   | 4.27 |
| PPLCNet_x2_5   | 5.39 |

**备注：** 精度类型为 FP32，推理过程使用 MKLDNN。

<a name="1.4.2"></a>

#### 1.4.2 基于 V100 GPU 的预测速度

| Models        | Latency(ms)<br>bs=1 | Latency(ms)<br/>bs=4 | Latency(ms)<br/>bs=8 |
| :--: | :--:| :--: | :--: |
| PPLCNet_x0_25 | 0.72                         | 1.17                             | 1.71                           |
| PPLCNet_x0_35 | 0.69                         | 1.21                             | 1.82                           |
| PPLCNet_x0_5  | 0.70                         | 1.32                             | 1.94                           |
| PPLCNet_x0_75 | 0.71                         | 1.49                             | 2.19                           |
| PPLCNet_x1_0  | 0.73                         | 1.64                             | 2.53                           |
| PPLCNet_x1_5  | 0.82                         | 2.06                             | 3.12                           |
| PPLCNet_x2_0  | 0.94                         | 2.58                             | 4.08                           |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT。

<a name="1.4.3"></a>

#### 1.4.3 基于 SD855 的预测速度

| Models        | Latency(ms)<br>bs=1, thread=1 | Latency(ms)<br/>bs=1, thread=2 | Latency(ms)<br/>bs=1, thread=4 |
| :--: | :--: | :--: | :--: |
| PPLCNet_x0_25 | 2.30                             | 1.62                              | 1.32                              |
| PPLCNet_x0_35 | 3.15                             | 2.11                              | 1.64                              |
| PPLCNet_x0_5  | 4.27                             | 2.73                              | 1.92                              |
| PPLCNet_x0_75 | 7.38                             | 4.51                              | 2.91                              |
| PPLCNet_x1_0  | 10.78                            | 6.49                              | 3.98                              |
| PPLCNet_x1_5  | 20.55                            | 12.26                             | 7.54                              |
| PPLCNet_x2_0  | 33.79                            | 20.17                             | 12.10                             |
| PPLCNet_x2_5  | 49.89                            | 29.60                             | 17.82                             |

**备注：** 精度类型为 FP32。

<a name="2"></a>  

## 2. 模型快速体验

<a name="2.1"></a>  

### 2.1 安装 paddlepaddle

- 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- 您的机器是CPU，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="2.2"></a>  

### 2.2 安装 paddleclas

使用如下命令快速安装 paddleclas

```  
pip3 install paddleclas
```

<a name="2.3"></a>

### 2.3 预测

* 在命令行中使用 PPLCNet_x1_0 的权重快速预测

```bash
paddleclas --model_name=PPLCNet_x1_0  --infer_imgs="docs/images/inference_deployment/whl_demo.jpg"
```

结果如下：
```
>>> result
class_ids: [8, 7, 86, 81, 85], scores: [0.91347, 0.03779, 0.0036, 0.00117, 0.00112], label_names: ['hen', 'cock', 'partridge', 'ptarmigan', 'quail'], filename: docs/images/inference_deployment/whl_demo.jpg
Predict complete!
```  

**备注**： 更换 PPLCNet 的其他 scale 的模型时，只需替换 `model_name`，如将此时的模型改为 `PPLCNet_x2_0` 时，只需要将 `--model_name=PPLCNet_x1_0` 改为 `--model_name=PPLCNet_x2_0` 即可。  

* 在 Python 代码中预测
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='PPLCNet_x1_0')
infer_imgs='docs/images/deployment/whl_demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

**备注**：`PaddleClas.predict()` 为可迭代对象（`generator`），因此需要使用 `next()` 函数或 `for` 循环对其迭
代调用。每次调用将以 `batch_size` 为单位进行一次预测，并返回预测结果。返回结果示例如下：

```
>>> result
[{'class_ids': [8, 7, 86, 81, 85], 'scores': [0.91347, 0.03779, 0.0036, 0.00117, 0.00112], 'label_names': ['hen', 'cock', 'partridge', 'ptarmigan', 'quail'], 'filename': 'docs/images/inference_deployment/whl_demo.jpg'}]
```

<a name="3"></a>

## 3. 模型训练、评估和预测

<a name="3.1"></a>  

### 3.1 环境配置

* 安装：请先参考 [Paddle 安装教程](../installation/install_paddle.md) 以及 [PaddleClas 安装教程](../../installation.md) 配置 PaddleClas 运行环境。

<a name="3.2"></a>

### 3.2 数据准备

请在[ImageNet 官网](https://www.image-net.org/)准备 ImageNet-1k 相关的数据。

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，将下载好的数据命名为 `ILSVRC2012` ，存放于此。 `ILSVRC2012` 目录中具有以下数据：

```
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
├── train_list.txt
...
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
├── val_list.txt
```

其中 `train/` 和 `val/` 分别为训练集和验证集。`train_list.txt` 和 `val_list.txt` 分别为训练集和验证集的标签文件。
<<<<<<< f18496291fdbc9ca57ece7790b00f456b68a1f68:docs/zh_CN/models/ImageNet1k/PP-LCNet.md

**备注：**

* 关于 `train_list.txt`、`val_list.txt`的格式说明，可以参考[PaddleClas分类数据集格式说明](../../training/single_label_classification/dataset.md#1-数据集格式说明) 。

**备注：**

<a name="3.3"></a>

### 3.3 模型训练

<a name="3.3.1"></a>

#### 3.3.1 训练 ImageNet

在 `ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml` 中提供了 PPLCNet_x1_0 训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml
```

**备注：**

* 当前精度最佳的模型会保存在 `output/PPLCNet_x1_0/best_model.pdparams`

#### 3.3.2 基于 ImageNet 权重微调

如果训练的不是 ImageNet 任务，而是其他任务时，需要更改配置文件和训练方法，详情可以参考：[模型微调](../../training/single_label_classification/finetune.md)。

<a name="3.4"></a>

### 3.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

其中 `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

<a name="3.5"></a>

### 3.5 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

输出结果如下：

```
[{'class_ids': [8, 7, 86, 81, 85], 'scores': [0.91347, 0.03779, 0.0036, 0.00117, 0.00112], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ptarmigan', 'quail']}]
```

**备注：**

* 这里`-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

* 默认是对 `docs/images/inference_deployment/whl_demo.jpg` 进行预测，此处也可以通过增加字段 `-o Infer.infer_imgs=xxx` 对其他图片预测。

* 默认输出的是 Top-5 的值，如果希望输出 Top-k 的值，可以指定`-o Infer.PostProcess.topk=k`，其中，`k` 为您指定的值。

* 默认的标签映射基于 ImageNet 数据集，如果改变数据集，需要重新指定`Infer.PostProcess.class_id_map_file`，该映射文件的制作方法可以参考`ppcls/utils/imagenet1k_label_list.txt`。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

当使用 Paddle Inference 推理时，加载的模型类型为 inference 模型。本案例提供了两种获得 inference 模型的方法，如果希望得到和文档相同的结果，请选择[直接下载 inference 模型](#6.1.2)的方式。

<a name="4.1.1"></a>

### 4.1.1 基于训练得到的权重导出 inference 模型

此处，我们提供了将权重和模型转换的脚本，执行该脚本可以得到对应的 inference 模型：

```bash
python3 tools/export_model.py \
    -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_infer
```
执行完该脚本后会在 `deploy/models/` 下生成 `PPLCNet_x1_0_infer` 文件夹，`models` 文件夹下应有如下文件结构：

```
├── PPLCNet_x1_0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="4.1.2"></a>

### 4.1.2 直接下载 inference 模型

[4.1.1 小节](#4.1.1)提供了导出 inference 模型的方法，此处也提供了该场景可以下载的 inference 模型，可以直接下载体验。

```
cd deploy/models
# 下载 inference 模型并解压
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar && tar -xf PPLCNet_x1_0_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── PPLCNet_x1_0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

<a name="4.2.1"></a>  

#### 4.2.1 预测单张图像

返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/ImageNet/ILSVRC2012_val_00000010.jpeg` 进行分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPLCNet_x1_0_infer
# 使用下面的命令使用 CPU 进行预测
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPLCNet_x1_0_infer -o Global.use_gpu=False
```

输出结果如下。

```
ILSVRC2012_val_00000010.jpeg:	class id(s): [153, 265, 204, 283, 229], score(s): [0.61, 0.11, 0.05, 0.03, 0.02], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'toy poodle', 'Lhasa, Lhasa apso', 'Persian cat', 'Old English sheepdog, bobtail']
```

<a name="4.2.2"></a>  

#### 4.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPLCNet_x1_0_infer -o Global.infer_imgs=images/ImageNet/
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
ILSVRC2012_val_00000010.jpeg:	class id(s): [153, 265, 204, 283, 229], score(s): [0.61, 0.11, 0.05, 0.03, 0.02], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'toy poodle', 'Lhasa, Lhasa apso', 'Persian cat', 'Old English sheepdog, bobtail']
ILSVRC2012_val_00010010.jpeg:	class id(s): [695, 551, 507, 531, 419], score(s): [0.11, 0.06, 0.03, 0.03, 0.03], label_name(s): ['padlock', 'face powder', 'combination lock', 'digital watch', 'Band Aid']
ILSVRC2012_val_00020010.jpeg:	class id(s): [178, 211, 209, 210, 236], score(s): [0.87, 0.03, 0.01, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'Chesapeake Bay retriever', 'German short-haired pointer', 'Doberman, Doberman pinscher']
ILSVRC2012_val_00030010.jpeg:	class id(s): [80, 23, 93, 81, 99], score(s): [0.87, 0.01, 0.01, 0.01, 0.00], label_name(s): ['black grouse', 'vulture', 'hornbill', 'ptarmigan', 'goose']
```

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应的部署工作。

<a name="5"></a>

## 5. 引用

如果你的论文用到了 PP-LCNet 的方法，请添加如下 cite：
```
@misc{cui2021pplcnet,
      title={PP-LCNet: A Lightweight CPU Convolutional Neural Network},
      author={Cheng Cui and Tingquan Gao and Shengyu Wei and Yuning Du and Ruoyu Guo and Shuilong Dong and Bin Lu and Ying Zhou and Xueying Lv and Qiwen Liu and Xiaoguang Hu and Dianhai Yu and Yanjun Ma},
      year={2021},
      eprint={2109.15099},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

