# RAM, RAM++ 图文标签模型

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

RAM以及RAM++（下文简称RAM类模型）主要用于标注类任务，其中两个模型的主要贡献为提出了集训练-推理-tag一体化的框架。其通过堆叠vision encoder以及text encoder，实现多种下游任务。核心方法包括：

1. 结合CLIP架构，提出 Image-Tag Recognition Decoder，Image-Text Alignment Encoder，Image-Tag Interaction Encoder，Image-Tag-Text Generation Decoder以及Generation Encoder 5个组件分别实现text-image对齐，text-tag对齐。
2. RAM++进一步使用大语言模型（large language model，LLM）的语义信息，提升text-image对齐的能力。

使用RAM类模型时，作者在多个分类任务上取得了最先进的结果：

| Model | BackBone   | Store Size   | Inference Prompt | CLIP | OpenImages-MAP |
|-------|------------|--------|------------------|------|----------------|
| RAM   | Swin-large | 5.63GB | LLM Tag Dec      |  VIT-base-patch16-224 | 82.2           |
| RAM++ | Swin-base  | 3.01GB | LLM Tag Dec      |  VIT-base-patch16-224 | 86.6           |
注：LLM Tag Dec表示基于LLM改写的文本tag。例如给定prompt："A photo of a cat" 对应LLM tag Dec为："Cat is a small general with sofa".

`PaddleClas` paddleclas分别实现了基于不同backbone的RAM类模型:
```yaml
# model architecture
Arch:
  name: ram_plus
  vit: swin_l
  vit_grad_ckpt: False
  vit_ckpt_layer: 0
  image_size: 384
  prompt: 'a picture of '
  med_config: 'ppcls/configs/ram/ram_bert.yaml'
  delete_tag_index: []
  tag_list: 'ppcls/utils/ram/ram_tag_list.txt'
  tag_list_chinese: 'ppcls/utils/ram/ram_tag_list_chinese.txt'
  clip_pretraind: ./ViT-B-32.pdparams #for CLIP a necessary part for training ram
  clip_version: 'vit-b-32-224'
  q2l_config: 'ppcls/configs/ram/ram_q2l.yaml'
  ram_class_threshold_path: 'ppcls/utils/RAM/ram_tag_list_threshold.txt'
  stage: train
  threshold: 0.68
```
参数注释：
  - name  模型参数，使用RAM模型可以指定为ram，使用RAM++模型可以指定为ram_plus，默认为ram 
  - vit  视觉主干网络参数，包括 vit：vision transformer，swin_b： swin base模型，swin_l, swin large模型
  - image_size  图片分辨率
  - prompt RAM训练时所使用的文本提示前缀
  - med_config RAM类模型所使用的Bert模型配置文件，默认配置路径：'ppcls/configs/ram/config_bert.yaml'
  - delete_tag_index 屏蔽tag所用参数，例如传递[1,3,2]则表示屏蔽index为1，2，3的tag标签
  - tag_list  英文tag标签文件路劲，默认ppcls/utils/RAM/ram_tag_list.txt
  - tag_list_chinese 中文tag标签文件路劲，默认ppcls/utils/RAM/ram_tag_list.txt
  - clip_version  所使用的CLIP结构，默认 vit-b-32-224
  - clip_pretraind 训练所使用的CLIP预训练参数路径，当需要训练RAM类模型时，不能为None
  - q2l_config  基于bert 的text-tag alignment encoder模型配置文件默认  ppcls/configs/ram/config_q2l.yaml 
  - ram_class_threshold_path  tag生成阈值文件默认ppcls/utils/RAM/ram_tag_list_threshold.txt
  - stage  指定RAM，RAM++模型是否进行训练，stage = train表示需要训练，训练时clip_pretraind不能为None，stage = eval表示无需训练
  - threshold 输出TAG所需的阈值数值，表示当该tag对应概率大于该值，则认为属于该tag
注意，RAM类模型的推理和训练，需要使用tools/train_multimodal.py, tools/infer_multimodal.py 以及predict_multimodal.py接口，支持多模态输入的动态图训练，推理以及静态图推理。

<a name="2"></a>
## 2. 数据和模型准备

* 前往官方[repo](https://github.com/xinyu1205/recognize-anything/tree/main)下载对应数据集json文件。同时按照json文件目录格式，准备相应的数据。目录格式为：
```json
{
    {
        "image_path": "visual-genome/VG_100K/1.jpg", 
        "caption": ["trees line the sidewalk"],
        "union_label_id": [4480], 
        "parse_label_id": [[4253, 2461, 2966]]
    }
}
```
参数注释：
  - image_path  数据集路径
  - caption  对应图片标注
  - union_label_id 标注对应id
  - parse_label_id 将标注仅需名词化后，结果对应的id。例如将"trees line the sidewalk"名词化得到 "trees" "line"以及"sidewalk"，其对应的id分别是4253, 2461, 2966
其中务必保证数据集文件路径符合image_path
* 本文档中，为RAM类模型提供了统一的动态训练以及推理配置文件，结构如下：
```yaml
# global configs
Global:
  checkpoints: null
  pretrained_model: "ram.pdparams" # pretrain model for ram and ram plus, default random initilize
  output_dir: ./output/
  device: gpu
  save_interval: 1
  eval_during_train: True
  eval_interval: 1
  epochs: 120
  print_batch_step: 10
  use_visualdl: False
  # used for static mode and model export
  image_shape: [3, 384, 384]
  save_inference_dir: ./inference


# mixed precision
AMP:
  use_amp: True
  use_fp16_test: False
  scale_loss: 128.0
  use_dynamic_loss_scaling: True
  use_promote: False
  # O1: mixed fp16, O2: pure fp16
  level: O1


# model architecture
Arch:
  name: ram
  vit: swin_l
  vit_grad_ckpt: False
  vit_ckpt_layer: 0
  image_size: 384
  prompt: 'a picture of '
  med_config: 'ppcls/configs/ram/ram_bert.yaml'
  delete_tag_index: []
  tag_list: 'ppcls/utils/ram/ram_tag_list.txt'
  tag_list_chinese: 'ppcls/utils/ram/ram_tag_list_chinese.txt'
  clip_pretraind: ./ViT-B-32.pdparams #for CLIP a necessary part for training ram
  clip_version: 'vit-b-32-224'
  q2l_config: 'ppcls/configs/ram/ram_q2l.yaml'
  ram_class_threshold_path: 'ppcls/utils/RAM/ram_tag_list_threshold.txt'
  stage: train
  threshold: 0.68
 
# loss function config for traing/eval process
Loss:
  Train:
    - RAMLoss:
        weight: 1.0
  Eval:
    - RAMLoss:
        weight: 1.0

Optimizer:
  name: AdamW
  beta1: 0.9
  beta2: 0.999
  epsilon: 1e-8
  weight_decay: 0.05
  layerwise_decay: 0.6
  filter_bias_and_bn: True
  lr:
    name: Cosine
    learning_rate: 0.0004
    eta_min: 1e-6
    warmup_epoch: 10
    warmup_start_lr: 5e-7

# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: RAMPretrainDataset
      ann_file: [./visual-genome/vg_ram.json]
      transform_ops_ram:
        - ResizeImage:
            size: 384
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
      transform_ops_clip:        
        - ResizeImage:
            resize_short: 224
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.48145466, 0.4578275, 0.40821073]
            std: [0.26862954, 0.26130258, 0.27577711]
            order: ''
        - ToCHWImage:

    sampler:
      name: DistributedBatchSampler
      batch_size: 52
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

  Eval:
    dataset: 
      name: RAMPretrainDataset
      ann_file: [./visual-genome/vg_ram.json]
      transform_ops_ram:
        - ResizeImage:
            size: 384
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
      transform_ops_clip:        
        - ResizeImage:
            resize_short: 224
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.48145466, 0.4578275, 0.40821073]
            std: [0.26862954, 0.26130258, 0.27577711]
            order: ''
        - ToCHWImage:

    sampler:
      name: DistributedBatchSampler
      batch_size: 52
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

Infer:
  infer_imgs: docs/images/inference_deployment/whl_demo.jpg
  batch_size: 1
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 384
    - CropImage:
        size: 384
    - NormalizeImage:
          scale: 1.0/255.0
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: ''
    - ToCHWImage:
  PostProcess:
    name: RamOutPut
    language: "cn"
    tag_list: "ppcls/utils/RAM/ram_tag_list.txt"
    tag_list_chinese: "ppcls/utils/RAM/ram_tag_list_chinese.txt"
    ram_class_threshold_path: "ppcls/utils/RAM/ram_tag_list_threshold.txt"


```
用户可以根据自身需求，更改相应配置。注意arch参数请参照本文档。

## 3. 模型训练
以RAM为例：
```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train_multimodal.py \
        -c ./ppcls/configs/ram/RAM.yaml
# 单卡
python3 tools/train_multimodal.py \
        -c ./ppcls/configs//ram/RAM.yaml
```
以RAM++为例：
```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train_multimodal.py \
        -c ./ppcls/configs/ram/RAM_plus.yaml
# 单卡
python3 tools/train_multimodal.py \
        -c ./ppcls/configs//ram/RAM_plus.yaml
```


<a name="4"></a>
## 4. 模型预测

```bash
python3 tools/infer_multimodal.py \
    -c ./ppcls/configs/ram/RAM.yaml \
    -o Global.pretrained_model="./output/ram/best_model"
```

得到类似下面的输出：
```
{'class_ids': [[[593], [871], [998], [2071], [3336], [3862], [4389]]], 'scores': [[[0.9708361625671387], [0.9998403787612915], [0.9122695922851562], 
[0.8888279795646667], [0.8671568036079407], [0.8900104761123657], [0.811939001083374]]], 'label_names': ['棕色 | 鸡 | 公鸡 | 母鸡  | 红色 | 站/矗立/摊位 | 走 ']}
```

<a name="5"></a>
## 5. 基于预测引擎预测

<a name="5.1"></a>
### 5.1 导出 inference model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/ram/RAM.yaml \
    -o Global.pretrained_model="./output/ram/"
```
inference model 的路径默认在当前路径下 `./inference`
`./inference` 文件夹下应有如下文件结构：

```
├── inference
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="5.2"></a>

### 5.2 基于 Python 预测引擎推理

切换到depoly目录下，并且使用deploy中的脚本进行推理前需要确认paddleclas为非本地安装, 如不是请进行切换，不然会出现包的导入错误。 

```shell
# 本地安装
pip install -e .
# 非本地安装
python setup.py install

# 进入deploy目录下
cd deploy
```

<a name="5.2.1"></a>  

#### 5.2.1 预测单张图像

运行下面的命令，对图像 `docs/images/inference_deployment/whl_demo.jpg` 进行分类。

```shell
# linux使用`python3`，windows使用`python (-m)`来执行脚本
# 使用下面的命令使用 GPU 进行预测
python3 python/predict_multimodal.py \
    -c deploy/configs/inference_ram.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg 
# 使用下面的命令使用 CPU 进行预测
#更改 `config_infer.yaml` 配置文件后
python3 python/predict_multimodal.py \
    -c deploy/configs/inference_ram.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg 
```

输出结果如下：

```
whl_demo.jpg-class_ids:  [[[593], [871], [998], [2071], [3336], [3862], [4389]]],
whl_demo.jpg-scores: [[[0.9708361625671387], [0.9998403787612915], [0.9122695922851562], 
[0.8888279795646667], [0.8671568036079407], [0.8900104761123657], [0.811939001083374]]], 
whl_demo.jpg-label_names: ['棕色 | 鸡 | 公鸡 | 母鸡  | 红色 | 站/矗立/摊位 | 走 ']
```


<a name="6"></a>
## 6. 引用
```
@article{huang2023inject,
  title={Inject Semantic Concepts into Image Tagging for Open-Set Recognition},
  author={Huang, Xinyu and Huang, Yi-Jie and Zhang, Youcai and Tian, Weiwei and Feng, Rui and Zhang, Yuejie and Xie, Yanchun and Li, Yaqian and Zhang, Lei},
  journal={arXiv preprint arXiv:2310.15200},
  year={2023}
}

@article{zhang2023recognize,
  title={Recognize Anything: A Strong Image Tagging Model},
  author={Zhang, Youcai and Huang, Xinyu and Ma, Jinyu and Li, Zhaoyang and Luo, Zhaochuan and Xie, Yanchun and Qin, Yuzhuo and Luo, Tong and Li, Yaqian and Liu, Shilong and others},
  journal={arXiv preprint arXiv:2306.03514},
  year={2023}
}
```