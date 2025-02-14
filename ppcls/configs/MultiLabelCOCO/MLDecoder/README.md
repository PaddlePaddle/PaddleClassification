# ML-Decoder多标签分类

## 目录

* [1. 模型介绍](#1)
* [2. 数据和模型准备](#2)
* [3. 模型训练](#3)
* [4. 模型评估](#4)
* [5. 模型预测](#5)
* [6. 基于预测引擎预测](#6)
  * [6.1 导出 inference model](#6.1)
  * [6.2 基于 Python 预测引擎推理](#6.2)
* [7. 引用](#7)

<a name="1"></a>
## 1. 模型介绍

ML-Decoder是一种新的基于注意力的分类头，它通过查询来预测类别标签的存在，并且比全局平均池化能够更好地利用空间数据。ML-Decoder的特点有以下几点：

1. ML-Decoder重新设计了解码器的架构，并且使用了一种新颖的分组解码方案，使得ML-Decoder具有高效性和可扩展性，可以很好地处理数千个类别的分类任务。
2. ML-Decoder具有一致性的速度-准确度权衡，相比于使用更大的主干网络，ML-Decoder能够提供更好的性能。
3. ML-Decoder也具有多功能性，它可以作为各种分类头的替代方案，并且在使用词语查询时能够泛化到未见过的类别。通过使用新颖的查询增强方法，ML-Decoder的泛化能力进一步提高。

使用ML-Decoder，作者在多个分类任务上取得了最先进的结果：
1. 在MS-COCO多标签分类上，达到了91.4%的mAP；
2. 在NUS-WIDE零样本分类上，达到了31.1%的ZSL mAP；
3. 在ImageNet单标签分类上，使用普通的ResNet50主干网络，达到了80.7%的新高分，而没有使用额外的数据或蒸馏。

`PaddleClas` 目前支持在单标签分类和多标签分类任务中使用ML-Decoder, 且使用方式非常简单方便，须在配置文件的Arch作用域下设置use_ml_decoder=True 并给出对ML-Decoder的参数配置即可:
```yaml
# model architecture
Arch:
  name: ResNet101
  class_num: 80
  pretrained: True
  # use ml-decoder head to replace avg_pool and fc
  use_ml_decoder: True

# ml-decoder head
MLDecoder:
  query_num: 80 # default: 80, query_num <= class_num
  in_channels: 2048
  # optional args
  # class_num: 80
  # remove_layers: ['avg_pool', 'flatten']
  # replace_layer: 'fc'
```
注意：
1. 当所选择的Backbone无法调取class_num属性时，MLDecoder需要在配置文件中手动添加`class_num`属性。
2. 实际使用时可根据对应Backbone中的实际的层的名称来修改`remove_layers`和`replace_layer`参数。 Backbone中的实际的层的名称可根据选取的主干模型名在`ppcls/arch/backbone`路径下查找对应的实现代码，并查看改模型相应输出层的名称。

下面是选择`RepVGG_A0`为例， 对应查看`ppcls/arch/backbone/model_zoo/repvgg.py`, 适应性添加或修改`class_num`, `remove_layers`和`replace_layer`后的MLDecoder配置示例：
```yaml
# model architecture
Arch:
  name: RepVGG_A0
  class_num: 80
  pretrained: True
  # use ml-decoder head to replace avg_pool and fc
  use_ml_decoder: True

# ml-decoder head
MLDecoder:
  query_num: 80 # default: 80, query_num <= class_num
  in_channels: 1280
  # optional args
  class_num: 80
  remove_layers: ['gap']
  replace_layer: 'linear'
```

开发者可以自行尝试使用不同的主干模型来结合ML-Decoder。

目前使用多类Backbone结合ML-Decoder在COCO2017的多标签分类任务上的性能指标如下：

|        Model         | Backbone  | Resolution | mAP |                   Links                   |
|:--------------------:|:---------:|:----------:|:---:|:-----------------------------------------:|
| PP-LCNet_x1_0_ml_decoder_448 | PP-LCNet_x1_0 |  448x448   |  77.96%  | [config](./PP-LCNet_x1_0_ml_decoder_448.yaml) |
| PP-HGNetV2-B0_ml_decoder_448 | PP-HGNetV2-B0 |  448x448   |  80.98%  | [config](./PP-HGNetV2-B0_ml_decoder_448.yaml) |
| PP-HGNetV2-B4_ml_decoder_448 | PP-HGNetV2-B4 |  448x448   |  87.96%   | [config](./PP-HGNetV2-B4_ml_decoder_448.yaml) |
| PP-HGNetV2-B6_ml_decoder_448 | PP-HGNetV2-B6 |  448x448   |  91.06%  | [config](./PP-HGNetV2-B6_ml_decoder_448.yaml) |
| ResNet50_ml_decoder_448 | ResNet50 |  448x448   |  83.42%  | [config](./ResNet50_ml_decoder_448.yaml) |
| ResNet101_ml_decoder | ResNet101 |  448x448   |  91.40%   | [config](./ResNet101_ml_decoder_448.yaml) |
| CLIP_vit_base_patch16_448_ml_decoder_448 | CLIP_vit_base_patch16_448 |  448x448   |  89.15%   | [config](./CLIP_vit_base_patch16_448_ml_decoder_448.yaml) |

基于 [COCO2017](https://cocodataset.org/) 数据集，如下将介绍添加ml-decoder进行多标签分类的训练、评估、预测的过程。请首先安装 PaddlePaddle 和 PaddleClas，具体安装步骤可详看 [环境准备](../installation.md)。



<a name="2"></a>
## 2. 数据和模型准备

* 进入 `PaddleClas` 目录。

```
cd path_to_PaddleClas
```

* 创建并进入 `dataset/COCO2017` 目录，下载并解压 COCO2017 数据集。

```shell
mkdir dataset/COCO2017 && cd dataset/COCO2017
wget http://images.cocodataset.org/zips/train2017.zip -O t.zip && unzip t.zip -d . && rm t.zip
wget http://images.cocodataset.org/zips/val2017.zip -O t.zip && unzip t.zip -d . && rm t.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O t.zip && unzip t.zip -d . && rm t.zip
```

* 返回 `PaddleClas` 根目录


```shell
cd ../../
# 转换训练集并生成`COCO2017_labels.txt`
python3 ./ppcls/utils/create_coco_multilabel_lists.py \
        --dataset_dir dataset/COCO2017 \
        --image_dir train2017 \
        --anno_path annotations/instances_train2017.json \
        --save_name multilabel_train_list --save_label_name
# 转换测试集
python3 ./ppcls/utils/create_coco_multilabel_lists.py \
        --dataset_dir dataset/COCO2017 \
        --image_dir val2017 \
        --anno_path annotations/instances_val2017.json \
        --save_name multilabel_val_list
```

<a name="3"></a>
## 3. 模型训练

```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet101_ml_decoder_448.yaml
# 单卡
python3 tools/train.py \
        -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet101_ml_decoder_448.yaml
```

**注意:**
1. 目前多标签分类的损失函数默认使用`MultiLabelAsymmetricLoss`。
2. 目前多标签分类的评估指标默认使用`MultiLabelMAP(integral)`。

<a name="4"></a>

## 4. 模型评估

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet101_ml_decoder_448.yaml \
    -o Global.pretrained_model="./output/ResNet101_ml_decoder_448/best_model"
```

<a name="5"></a>
## 5. 模型预测

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet101_ml_decoder_448.yaml \
    -o Global.pretrained_model="./output/ResNet101_ml_decoder_448/best_model"
```

得到类似下面的输出：
```
[{'class_ids': [0, 2, 7, 24, 25, 26, 33, 56], 'scores': [0.99998, 0.52104, 0.51953, 0.59292, 0.64329, 0.63605, 0.99994, 0.7054], 'label_names': ['person', 'car', 'truck', 'backpack', 'umbrella', 'handbag', 'kite', 'chair'], 'file_name': 'deploy/images/coco_000000570688.jpg'}]
```

<a name="6"></a>
## 6. 基于预测引擎预测

<a name="6.1"></a>
### 6.1 导出 inference model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/MultiLabelCOCO/MLDecoder/ResNet101_ml_decoder_448.yaml \
    -o Global.pretrained_model="./output/ResNet101_ml_decoder_448/best_model"
```
inference model 的路径默认在当前路径下 `./inference`
`./inference` 文件夹下应有如下文件结构：

```
├── inference
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="6.2"></a>

### 6.2 基于 Python 预测引擎推理

切换到depoly目录下，并且使用deploy中的脚本进行推理前需要确认paddleclas为非本地安装, 如不是请进行切换，不然会出现包的导入错误。 

```shell
# 本地安装
pip install -e .
# 非本地安装
python setup.py install

# 进入deploy目录下
cd deploy
```

<a name="6.2.1"></a>  

#### 6.2.1 预测单张图像

运行下面的命令，对图像 `./images/coco_000000570688.jpg` 进行分类。

```shell
# linux使用`python3`，windows使用`python (-m)`来执行脚本
# 使用下面的命令使用 GPU 进行预测
python3 python/predict_cls.py \
    -c configs/inference_cls_multilabel.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=images/coco_000000570688.jpg \
    -o PostProcess.MultiLabelThreshOutput.class_id_map_file=../ppcls/utils/COCO2017_label_list.txt 
# 使用下面的命令使用 CPU 进行预测
python3 python/predict_cls.py \
    -c configs/inference_cls_multilabel.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=images/coco_000000570688.jpg \
    -o PostProcess.MultiLabelThreshOutput.class_id_map_file=../ppcls/utils/COCO2017_label_list.txt \
    -o Global.use_gpu=False
```

输出结果如下：

```
coco_000000570688.jpg:  class id(s): [0, 2, 3, 4, 7, 9, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 38, 39, 45, 46, 47, 48, 49, 51, 52, 53, 54, 57, 58, 60, 61, 62, 63, 64, 65, 67, 69, 70, 71, 72, 73, 75], score(s): [0.84, 0.68, 0.93, 0.54, 0.74, 0.90, 0.56, 0.60, 0.63, 0.77, 0.64, 0.70, 0.94, 0.82, 0.99, 0.71, 0.86, 0.81, 0.81, 0.65, 0.65, 0.92, 0.67, 0.53, 0.83, 0.63, 0.58, 0.52, 0.83, 0.55, 0.92, 0.72, 0.74, 0.59, 0.82, 0.50, 0.62, 0.77, 0.87, 0.64, 0.84, 0.67], label_name(s): ['person', 'car', 'motorcycle', 'airplane', 'truck', 'traffic light', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie', 'suitcase', 'frisbee', 'skis', 'kite', 'tennis racket', 'bottle', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'couch', 'potted plant', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'cell phone', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'vase']
```

<a name="6.2.2"></a>  

#### 6.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# linux使用`python3`，windows使用`python (-m)`来执行脚本
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3 python/predict_cls.py \
    -c configs/inference_cls_multilabel.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o PostProcess.MultiLabelThreshOutput.class_id_map_file=../ppcls/utils/COCO2017_label_list.txt \
    -o Global.infer_imgs=images/ImageNet/
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
ILSVRC2012_val_00000010.jpeg:   class id(s): [0, 2, 3, 7, 9, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 38, 39, 40, 41, 45, 46, 47, 48, 49, 52, 53, 54, 58, 60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 73, 75], score(s): [0.80, 0.58, 0.89, 0.74, 0.86, 0.66, 0.56, 0.60, 0.81, 0.64, 0.73, 0.94, 0.75, 0.99, 0.70, 0.86, 0.78, 0.63, 0.57, 0.76, 0.66, 0.60, 0.94, 0.65, 0.90, 0.63, 0.52, 0.79, 0.50, 0.93, 0.72, 0.70, 0.60, 0.83, 0.61, 0.75, 0.86, 0.67, 0.87, 0.64], label_name(s): ['person', 'car', 'motorcycle', 'truck', 'traffic light', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie', 'suitcase', 'frisbee', 'skis', 'kite', 'tennis racket', 'bottle', 'wine glass', 'cup', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'hot dog', 'pizza', 'donut', 'potted plant', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'vase']
ILSVRC2012_val_00010010.jpeg:   class id(s): [0, 2, 3, 6, 7, 8, 9, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 38, 39, 40, 45, 46, 47, 48, 49, 51, 52, 53, 54, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 75], score(s): [0.78, 0.66, 0.93, 0.54, 0.77, 0.50, 0.86, 0.69, 0.63, 0.50, 0.78, 0.56, 0.71, 0.93, 0.78, 0.99, 0.64, 0.85, 0.80, 0.53, 0.85, 0.71, 0.66, 0.96, 0.70, 0.62, 0.85, 0.58, 0.57, 0.57, 0.78, 0.50, 0.92, 0.64, 0.73, 0.71, 0.77, 0.53, 0.66, 0.52, 0.73, 0.87, 0.69, 0.85, 0.66], label_name(s): ['person', 'car', 'motorcycle', 'train', 'truck', 'boat', 'traffic light', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie', 'suitcase', 'frisbee', 'skis', 'kite', 'tennis racket', 'bottle', 'wine glass', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'couch', 'potted plant', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'vase']
ILSVRC2012_val_00020010.jpeg:   class id(s): [0, 2, 3, 7, 9, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 38, 39, 40, 41, 45, 46, 47, 48, 49, 51, 52, 53, 54, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 75], score(s): [0.85, 0.62, 0.94, 0.69, 0.89, 0.56, 0.56, 0.62, 0.77, 0.67, 0.71, 0.93, 0.78, 0.99, 0.65, 0.86, 0.75, 0.57, 0.60, 0.76, 0.66, 0.58, 0.95, 0.73, 0.50, 0.88, 0.63, 0.59, 0.62, 0.84, 0.59, 0.82, 0.74, 0.74, 0.62, 0.86, 0.53, 0.53, 0.60, 0.73, 0.88, 0.65, 0.85, 0.70], label_name(s): ['person', 'car', 'motorcycle', 'truck', 'traffic light', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie', 'suitcase', 'frisbee', 'skis', 'kite', 'tennis racket', 'bottle', 'wine glass', 'cup', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'donut', 'couch', 'potted plant', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'vase']
ILSVRC2012_val_00030010.jpeg:   class id(s): [0, 2, 3, 4, 7, 9, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 38, 39, 40, 41, 45, 46, 47, 48, 49, 51, 52, 53, 58, 60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 73, 75], score(s): [0.82, 0.60, 0.92, 0.54, 0.67, 0.87, 0.57, 0.63, 0.57, 0.84, 0.70, 0.80, 0.92, 0.82, 0.99, 0.72, 0.86, 0.80, 0.59, 0.55, 0.84, 0.73, 0.60, 0.94, 0.75, 0.53, 0.89, 0.51, 0.84, 0.56, 0.90, 0.87, 0.67, 0.70, 0.85, 0.59, 0.82, 0.91, 0.62, 0.89, 0.67], label_name(s): ['person', 'car', 'motorcycle', 'airplane', 'truck', 'traffic light', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie', 'suitcase', 'frisbee', 'skis', 'kite', 'tennis racket', 'bottle', 'wine glass', 'cup', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'hot dog', 'pizza', 'potted plant', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'vase']
```

<a name="7"></a>
## 7. 引用
```
@misc{ridnik2021mldecoder,
      title={ML-Decoder: Scalable and Versatile Classification Head}, 
      author={Tal Ridnik and Gilad Sharir and Avi Ben-Cohen and Emanuel Ben-Baruch and Asaf Noy},
      year={2021},
      eprint={2111.12933},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```