简体中文 | [English](../../../en/image_recognition_pipeline/feature_extraction_en.md)
# 特征提取

## 目录

- [特征提取](#特征提取)
  - [目录](#目录)
  - [1. 摘要](#1-摘要)
  - [2. 介绍](#2-介绍)
  - [3. 方法](#3-方法)
      - [3.1 Backbone](#31-backbone)
      - [3.2 Neck](#32-neck)
      - [3.3 Head](#33-head)
      - [3.4 Loss](#34-loss)
      - [3.5 Data Augmentation](#35-data-augmentation)
  - [4. 实验部分](#4-实验部分)
  - [5. 自定义特征提取](#5-自定义特征提取)
    - [5.1 数据准备](#51-数据准备)
    - [5.2 模型训练](#52-模型训练)
    - [5.3 模型评估](#53-模型评估)
    - [5.4 模型推理](#54-模型推理)
      - [5.4.1 导出推理模型](#541-导出推理模型)
      - [5.4.2 获取特征向量](#542-获取特征向量)
  - [6. 总结](#6-总结)
  - [7. 参考文献](#7-参考文献)

<a name="1"></a>

## 1. 摘要

特征提取是图像识别中的关键一环，它的作用是将输入的图片转化为固定维度的特征向量，用于后续的[向量检索](../../deployment/PP-ShiTu/vector_search.md)。一个好的特征需要具备“相似度保持性”，即相似度高的图片对，其特征的相似度也比较高（特征空间中的距离比较近），相似度低的图片对，其特征相似度要比较低（特征空间中的距离比较远）。为此[Deep Metric Learning](../../algorithm_introduction/metric_learning.md)领域内提出了不少方法用以研究如何通过深度学习来获得具有强表征能力的特征。

<a name="2"></a>

## 2. 介绍

为了图像识别任务的灵活定制，我们将整个网络分为 Backbone、 Neck、 Head 以及 Loss 部分，整体结构如下图所示:
![](../../../images/feature_extraction_framework.png)
图中各个模块的功能为:

- **Backbone**: 用于提取输入图像初步特征的骨干网络，一般由配置文件中的 [Backbone](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L33-L37) 以及 [BackboneStopLayer](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L38-L39) 字段共同指定。
- **Neck**: 用以特征增强及特征维度变换。可以是一个简单的 FC Layer，用来做特征维度变换；也可以是较复杂的 FPN 结构，用以做特征增强，一般由配置文件中的 [Neck](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L40-L51) 字段指定。
- **Head**: 用来将 `Neck` 的输出 feature 转化为 logits，让模型在训练阶段能以分类任务的形式进行训练。除了常用的 FC Layer 外，还可以替换为 [CosMargin](../../../../ppcls/arch/gears/cosmargin.py), [ArcMargin](../../../../ppcls/arch/gears/arcmargin.py), [CircleMargin](../../../../ppcls/arch/gears/circlemargin.py) 等模块，一般由配置文件中的 [Head](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L52-L60) 字段指定。
- **Loss**: 指定所使用的 Loss 函数。我们将 Loss 设计为组合 loss 的形式，可以方便地将 Classification Loss 和 Metric learning Loss 组合在一起，一般由配置文件中的 [Loss](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L63-L77) 字段指定。

<a name="3"></a>

## 3. 方法

#### 3.1 Backbone

Backbone 部分采用了 [PP-LCNetV2_base](../../models/ImageNet1k/PP-LCNetV2.md)，其在 `PPLCNet_V1` 的基础上，加入了包括Rep 策略、PW 卷积、Shortcut、激活函数改进、SE 模块改进等多个优化点，使得最终分类精度与 `PPLCNet_x2_5` 相近，且推理延时减少了40%<sup>*</sup>。在实验过程中我们对 `PPLCNetV2_base` 进行了适当的改进，在保持速度基本不变的情况下，让其在识别任务中得到更高的性能，包括：去掉 `PPLCNetV2_base` 末尾的 `ReLU` 和 `FC`、将最后一个 stage(RepDepthwiseSeparable) 的 stride 改为1。


**注：** <sup>*</sup>推理环境基于 Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 硬件平台，OpenVINO 推理平台。

#### 3.2 Neck

Neck 部分采用了 [BN Neck](../../../../ppcls/arch/gears/bnneck.py)，对 Backbone 抽取得到的特征的每个维度进行标准化操作，减少了同时优化度量学习损失函数和分类损失函数的难度，加快收敛速度。

#### 3.3 Head

Head 部分选用 [FC Layer](../../../../ppcls/arch/gears/fc.py)，使用分类头将 feature 转换成 logits 供后续计算分类损失。

#### 3.4 Loss

Loss 部分选用 [Cross entropy loss](../../../../ppcls/loss/celoss.py) 和 [TripletAngularMarginLoss](../../../../ppcls/loss/tripletangularmarginloss.py)，在训练时以分类损失和基于角度的三元组损失来指导网络进行优化。我们基于原始的 TripletLoss (困难三元组损失)进行了改进，将优化目标从 L2 欧几里得空间更换成余弦空间，并加入了 anchor 与 positive/negtive 之间的硬性距离约束，让训练与测试的目标更加接近，提升模型的泛化能力。详细的配置文件见 [GeneralRecognitionV2_PPLCNetV2_base.yaml](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L63-77)。

#### 3.5 Data Augmentation

我们考虑到实际相机拍摄时目标主体可能出现一定的旋转而不一定能保持正立状态，因此我们在数据增强中加入了适当的 [随机旋转增强](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L117)，以提升模型在真实场景中的检索能力。

<a name="4"></a>

## 4. 实验部分

我们对原有的训练数据进行了合理扩充与优化，最终使用如下 17 个公开数据集的汇总：

| 数据集                 | 数据量  |  类别数  | 场景  |                                      数据集地址                                      |
| :--------------------- | :-----: | :------: | :---: | :----------------------------------------------------------------------------------: |
| Aliproduct             | 2498771 |  50030   | 商品  |      [地址](https://retailvisionworkshop.github.io/recognition_challenge_2020/)      |
| GLDv2                  | 1580470 |  81313   | 地标  |               [地址](https://github.com/cvdfoundation/google-landmark)               |
| VeRI-Wild              | 277797  |  30671   | 车辆  |                    [地址](https://github.com/PKU-IMRE/VERI-Wild)                     |
| LogoDet-3K             | 155427  |   3000   | Logo  |              [地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset)              |
| SOP                    |  59551  |  11318   | 商品  |              [地址](https://cvgl.stanford.edu/projects/lifted_struct/)               |
| Inshop                 |  25882  |   3997   | 商品  |            [地址](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)             |
| bird400                |  58388  |   400    | 鸟类  |          [地址](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)          |
| 104flows               |  12753  |   104    | 花类  |              [地址](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)              |
| Cars                   |  58315  |   112    | 车辆  |            [地址](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)            |
| Fashion Product Images |  44441  |    47    | 商品  | [地址](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) |
| flowerrecognition      |  24123  |    59    | 花类  |         [地址](https://www.kaggle.com/datasets/aymenktari/flowerrecognition)         |
| food-101               | 101000  |   101    | 食物  |         [地址](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)          |
| fruits-262             | 225639  |   262    | 水果  |            [地址](https://www.kaggle.com/datasets/aelchimminut/fruits262)            |
| inaturalist            | 265213  |   1010   | 自然  |           [地址](https://github.com/visipedia/inat_comp/tree/master/2017)            |
| indoor-scenes          |  15588  |    67    | 室内  |       [地址](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019)       |
| Products-10k           | 141931  |   9691   | 商品  |                       [地址](https://products-10k.github.io/)                        |
| CompCars               |  16016  |   431    | 车辆  |     [地址](http://​​​​​​http://ai.stanford.edu/~jkrause/cars/car_dataset.html​)      |
| **Total**              | **6M**  | **192K** |   -   |                                          -                                           |

最终的模型精度指标如下表所示:

| 模型                   | 延时(ms) | 存储(MB) | product<sup>*</sup> |      | Aliproduct |      | VeRI-Wild |      | LogoDet-3k |      | iCartoonFace |      | SOP      |      | Inshop   |      | gldv2    |      | imdb_face |      | iNat     |      | instre   |      | sketch   |      | sop      |      |
| :--------------------- | :------- | :------- | :------------------ | :--- | ---------- | ---- | --------- | ---- | ---------- | ---- | ------------ | ---- | -------- | ---- | -------- | ---- | -------- | ---- | --------- | ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- | ---- |
|                        |          |          | recall@1            | mAP  | recall@1   | mAP  | recall@1  | mAP  | recall@1   | mAP  | recall@1     | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1  | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1 | mAP  |
| PP-ShiTuV1_general_rec | 5.0      | 34       | 65.9                | 54.3 | 83.9       | 83.2 | 88.7      | 60.1 | 86.1       | 73.6 | 84.1         | 72.3 | 79.7     | 58.6 | 89.1     | 69.4 | 98.2     | 91.6 | 28.8      | 8.42 | 12.6     | 6.1  | 72.0     | 50.4 | 27.9     | 9.5  | 97.6     | 90.3 |
| PP-ShiTuV2_general_rec | 6.1      | 19       | 73.7                | 61.0 | 84.2       | 83.3 | 87.8      | 68.8 | 88.0       | 63.2 | 53.6         | 27.5 | 77.6     | 55.3 | 90.8     | 74.3 | 98.1     | 90.5 | 35.9      | 11.2 | 38.6     | 23.9 | 87.7     | 71.4 | 39.3     | 15.6 | 98.3     | 90.9 |

* product数据集是为了验证PP-ShiTu的泛化性能而制作的数据集，所有的数据都没有在训练和测试集中出现。该数据包含7个大类（化妆品、地标、红酒、手表、车、运动鞋、饮料），250个小类。测试时，使用250个小类的标签进行测试；sop数据集来自[GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval](https://arxiv.org/abs/2111.13122)，可视为“SOP”数据集的子集。
* 预训练模型地址：[general_PPLCNetV2_base_pretrained_v1.0.pdparams](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/PPShiTuV2/general_PPLCNetV2_base_pretrained_v1.0.pdparams)
* 采用的评测指标为：`Recall@1` 与 `mAP`
* 速度评测机器的 CPU 具体信息为：`Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz`
* 速度指标的评测条件为： 开启 MKLDNN, 线程数设置为 10

同时，我们也提供了服务端侧精度更高的识别模型，详细指标如下，使用方法可以参考 [PaddleX](https://github.com/PaddlePaddle/PaddleX)。

| 模型                             | recall@1 (%) | 模型存储大小 (M) |训练config|模型权重|
|---------------------------------|--------------|------------------|------------------|------------------|
| PP-ShiTuV2_rec_CLIP_vit_base    | 88.69        | 306.6 M          |[config](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_CLIP_vit_base.yaml)|[动态图](https://paddleclas.bj.bcebos.com/models/PP-ShiTu/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams)/[静态图](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b1_v2/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar)|
| PP-ShiTuV2_rec_CLIP_vit_large   | 91.03        | 1.05 G           |[config](../../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_CLIP_vit_large.yaml)|[静态图](https://paddleclas.bj.bcebos.com/models/PP-ShiTu/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams)/[静态图](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b1_v2/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar)|

* 注：以上精度指标为 AliProducts recall@1。 

<a name="5"></a>

## 5. 自定义特征提取

自定义特征提取，是指依据自己的任务，重新训练特征提取模型。

下面基于 `GeneralRecognitionV2_PPLCNetV2_base.yaml` 配置文件，介绍主要的四个步骤：1）数据准备；2）模型训练；3）模型评估；4）模型推理

<a name="5.1"></a>

### 5.1 数据准备

首先需要基于任务定制自己的数据集。数据集格式与文件结构详见 [数据集格式说明](../metric_learning/dataset.md)。

准备完毕之后还需要在配置文件中修改数据配置相关的内容, 主要包括数据集的地址以及类别数量。对应到配置文件中的位置如下所示：

- 修改类别数：
  ```yaml
  Head:
    name: FC
    embedding_size: *feat_dim
    class_num: 192612  # 此处表示类别数
    weight_attr:
      initializer:
        name: Normal
        std: 0.001
    bias_attr: False
  ```
- 修改训练数据集配置：
  ```yaml
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/     # 此处表示train数据集所在的目录
      cls_label_path: ./dataset/train_reg_all_data_v2.txt  # 此处表示train数据集对应标注文件的地址
      relabel: True
  ```
- 修改评估数据集中query数据配置：
  ```yaml
  Query:
    dataset:
      name: VeriWild
      image_root: ./dataset/Aliproduct/    # 此处表示query数据集所在的目录
      cls_label_path: ./dataset/Aliproduct/val_list.txt    # 此处表示query数据集对应标注文件的地址
  ```
- 修改评估数据集中gallery数据配置：
  ```yaml
  Gallery:
    dataset:
      name: VeriWild
      image_root: ./dataset/Aliproduct/    # 此处表示gallery数据集所在的目录
      cls_label_path: ./dataset/Aliproduct/val_list.txt   # 此处表示gallery数据集对应标注文件的地址
  ```

<a name="5.2"></a>

### 5.2 模型训练

模型训练主要包括启动训练和断点恢复训练的功能

- 单机单卡训练
  ```shell
  export CUDA_VISIBLE_DEVICES=0
  python3.7 tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml
  ```
- 单机多卡训练
  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
  tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml
  ```
**注意：**
配置文件中默认采用`在线评估`的方式，如果你想加快训练速度，可以关闭`在线评估`功能，只需要在上述命令的后面，增加 `-o Global.eval_during_train=False`。

训练完毕后，在 output 目录下会生成最终模型文件 `latest.pdparams`，`best_model.pdarams` 和训练日志文件 `train.log`。其中，`best_model` 保存了当前评测指标下的最佳模型，`latest` 用来保存最新生成的模型, 方便在任务中断的情况下从断点位置恢复训练。通过在上述训练命令的末尾加上`-o Global.checkpoints="path_to_resume_checkpoint"`即可从断点恢复训练，示例如下。

- 单机单卡断点恢复训练
  ```shell
  export CUDA_VISIBLE_DEVICES=0
  python3.7 tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o Global.checkpoints="output/RecModel/latest"
  ```
- 单机多卡断点恢复训练
  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
  tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o Global.checkpoints="output/RecModel/latest"
  ```

<a name="5.3"></a>

### 5.3 模型评估

除了训练过程中对模型进行的在线评估，也可以手动启动评估程序来获得指定的模型的精度指标。

- 单卡评估
  ```shell
  export CUDA_VISIBLE_DEVICES=0
  python3.7 tools/eval.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o Global.pretrained_model="output/RecModel/best_model"
  ```

- 多卡评估
  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
  tools/eval.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o  Global.pretrained_model="output/RecModel/best_model"
  ```
**注：** 建议使用多卡评估。该方式可以利用多卡并行计算快速得到全部数据的特征，能够加速评估的过程。

<a name="5.4"></a>

### 5.4 模型推理

推理过程包括两个步骤： 1）导出推理模型；2）模型推理以获取特征向量

#### 5.4.1 导出推理模型

首先需要将 `*.pdparams` 模型文件转换成 inference 格式，转换命令如下。
```shell
python3.7 tools/export_model.py \
-c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
-o Global.pretrained_model="output/RecModel/best_model"
```
生成的推理模型默认位于 `PaddleClas/inference` 目录，里面包含三个文件，分别为 `inference.pdmodel`、`inference.pdiparams`、`inference.pdiparams.info`。
其中`inference.pdmodel` 用来存储推理模型的结构, `inference.pdiparams` 和 `inference.pdiparams.info` 用来存储推理模型相关的参数信息。

#### 5.4.2 获取特征向量

使用上一步转换得到的 inference 格式模型，将输入图片转换为对应的特征向量，推理命令如下。

```shell
cd deploy
python3.7 python/predict_rec.py \
-c configs/inference_rec.yaml \
-o Global.rec_inference_model_dir="../inference"
```
得到的特征输出格式如下所示：

```log
wangzai.jpg:    [-7.82453567e-02  2.55877394e-02 -3.66694555e-02  1.34572461e-02
  4.39076796e-02 -2.34078392e-02 -9.49947070e-03  1.28221214e-02
  5.53947650e-02  1.01355985e-02 -1.06436480e-02  4.97181974e-02
 -2.21862812e-02 -1.75557341e-02  1.55848479e-02 -3.33278324e-03
 ...
 -3.40284109e-02  8.35561901e-02  2.10910216e-02 -3.27066667e-02]
```

在实际使用过程中，仅仅得到特征可能并不能满足业务需求。如果想进一步通过特征检索来进行图像识别，可以参照文档 [向量检索](../../deployment/PP-ShiTu/vector_search.md)。

<a name="6"></a>

## 6. 总结

特征提取模块作为图像识别中的关键一环，在网络结构的设计，损失函数的选取上有很大的改进空间。不同的数据集类型有各自不同的特点，如行人重识别、商品识别、人脸识别数据集的分布、图片内容都不尽相同。学术界根据这些特点提出了各种各样的方法，如PCB、MGN、ArcFace、CircleLoss、TripletLoss等，围绕的还是增大类间差异、减少类内差异的最终目标，从而有效地应对各种真实场景数据。

<a name="7"></a>

## 7. 参考文献

1. [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf)
2. [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
