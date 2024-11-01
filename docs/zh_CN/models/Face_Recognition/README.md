# 人脸识别模型

------


## 目录

- [1. 模型和应用场景介绍](##1.-模型和应用场景介绍)
- [2. 支持模型列列表](#2.-支持模型列表)
- [3. 模型快速体验](#3.-模型快速体验)
    - [3.1 安装paddlepaddle](#3.1-安装-paddlepaddle)
    - [3.2 安装PaddleClas](#3.2-安装-PaddleClas)
    - [3.3 模型训练与评估](#3.3-模型训练与评估)
      - [3.3.1 下载数据集](#3.3.1-下载数据集)
      - [3.3.2 模型训练](#3.3.2-模型训练)
      - [3.3.2 模型评估](#3.3.3-模型评估)


## 1. 模型和应用场景介绍

人脸识别模型通常以经过检测提取和关键点矫正处理的标准化人脸图像作为输入。人脸识别模型从这些图像中提取具有高度辨识性的人脸特征，以便供后续模块使用，如人脸匹配和验证等任务。PaddleClas 目前支持了基于 [ArcFace](https://arxiv.org/abs/1801.07698) 损失函数训练的人脸识别模型，包括 [MobileFaceNet](https://arxiv.org/abs/1804.07573) 和 ResNet50。同时也支持在 AgeDB-30、CFP-FP、LFW，CPLFW 和 CALFW 5个常用的人脸识别数据集上进行评估。

## 2. 支持模型列表

|模型|训练数据集|输出特征维度 | 损失函数 |Acc (%)<br>AgeDB-30/CFP-FP/LFW | 模型参数量(M) |模型下载|
|-|-|-|-|:-:|-|-|
| MobileFaceNet |MS1Mv3 |128 |ArcFace |96.28/96.71/99.58 | 0.99  |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/mobilefacenet.pdparams)|
| ResNet50      |MS1Mv3 |512 |ArcFace |98.12/98.56/99.77 | 25.56 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/resnet50_face.pdparams)|

**注：**

* 上述评估指标用到的数据集来自 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#validation-datasets) 提供的bin文件，评估流程也完全与该仓库对齐
* PaddleClas参照一般人脸识别模型的训练设置，将训练辨率设为112x112。原始的ResNet50模型在该分辨率下使用ArcFace损失函数进行训练时较难收敛，在此这我们参考 [insightface](https://github.com/deepinsight/insightface/blob/a1eb8523fbe50b0c0e39a9fa96d4e2a6936b46be/recognition/arcface_torch/backbones/iresnet.py#L39) 仓库中 IResNet50 的实现，将原始 ResNet50 的整体下采样倍率由原先的 32x 调整为 16x。详见训练配置文件[`Face_Recognition/FaceRecognition_ArcFace_ResNet50.yaml`](../../../../ppcls/configs/Face_Recognition/FaceRecognition_ArcFace_ResNet50.yaml)

## 3. 模型快速体验

### 3.1 安装 paddlepaddle

- 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- 您的机器是 CPU，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

### 3.2 安装 paddleclas

请确保已clone本项目，本地构建安装：

```
若之前安装有paddleclas，先使用以下命令卸载
python3 -m pip uninstall paddleclas
```

使用下面的命令构建:

```
cd path/to/PaddleClas
pip install -v -e .
```

### 3.3 模型训练与评估

#### 3.3.1 下载数据
执行以下命令下载人脸识别数据集MS1Mv3和人脸识别评估数据集，并解压到指定目录

```bash
cd path/to/PaddleClas
wget https://paddleclas.bj.bcebos.com/data/MS1M_v3.tar -P ./dataset/
tar -xf ./dataset/MS1M_v3.tar -C ./dataset/
```
成功执行后进入 `dataset/`目录，可以看到以下数据：

```bash
MS1M_v3
├── images              # 训练图像保存目录
│   ├── 00000001.jpg    # 训练图像文件
│   ├── 00000002.jpg    # 训练图像文件
│   │   ...
├── agedb_30.bin        # AgeDB-30 评估集文件
├── calfw.bin           # CALFW 评估集文件
├── cfp_fp.bin          # CFP-FP 评估集文件
├── cplfw.bin           # CPLFW 评估集文件
├── label.txt           # 训练集标注文件。每行给出图像的路径和人脸图像类别（人脸身份）id，使用空格分隔，内容举例：images/00000001.jpg 0
└── lfw.bin             # LFW 评估集文件
```
* 注：上述MS1Mv3数据集的训练图像和标签是从 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) 提供的`rec`格式的文件中恢复出来的，具体恢复过程可以参考 [AdaFace 仓库](https://github.com/mk-minchul/AdaFace/blob/master/README_TRAIN.md)。各评估集的`bin`文件也由 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#validation-datasets) 提供。

### 3.3.2 模型训练

在 `ppcls/configs/Face_Recognition` 目录中提供了训练配置，以 MobileFaceNet 为例，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/Face_Recognition/FaceRecognition_ArcFace_MobileFaceNet.yaml
```

 * 注：当前精度最佳的模型会保存在 `output/MobileFaceNet/best_model.pdparams`


### 3.3.3 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ppcls/configs/Face_Recognition/FaceRecognition_ArcFace_MobileFaceNet.yaml \
    -o Global.pretrained_model=output/MobileFaceNet/best_model
```

其中 `-o Global.pretrained_model="output/MobileFaceNet/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

