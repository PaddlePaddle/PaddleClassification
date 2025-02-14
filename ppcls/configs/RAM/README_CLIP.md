# CLIP完整模型向量落盘推理

## 目录

* [1. 综述](#1)
* [2. CLIP训练](#2)
* [3. CLIP动态、静态推理](#3)
* [5. CLIP向量落盘推理](#4)
* [6. 引用](#5)


<a name="1"></a>
## 1. 综述

CLIP为预训练的文本-图像大模型。其核心是利用对比损失函数（Constrative Loss），使用文本编码器（Text Encoder）和图像编码器(Image Encoder)将输入的本文和图像嵌入到相同的空间的中，通过衡量向量余玄相似度实现文本-图像匹配。本文所包含的内容主要为：1）CLIP模型的训练。2）基于CLIP，向量落盘，实现图文匹配。



```yaml
# model architecture
Arch:
  name: CLIP_vit_base_patch32_224_with_TextEncoder
  clip: "text"


```
参数注释：
  - name  模型参数，目前仅兼容vit为backbone的CLIP。
  - clip  输入类型参数，用于申明输入的类型。类型包括 "image"-图片以及"text"-文本。该参数仅作用于推理阶段。训练阶段此参数无实际意义。

<a name="2"></a>
## 2. CLIP训练

**数据集准备。** 训练数据结构遵照img2dataset[https://github.com/rom1504/img2dataset]

```yaml
# global configs
# global configs
Global:
  checkpoints: null
  pretrained_model: "ViT-B-32.pdparams" # pretrain model for ram and ram plus, default random initilize
  output_dir: ./output/
  device: cpu
  save_interval: 1
  eval_during_train: True
  eval_interval: 1
  epochs: 120
  print_batch_step: 1
  use_visualdl: False
  # used for static mode and model export
  image_shape: [3, 224, 224]
  export_shape: [77]
  export_type: "int64"
  save_inference_dir: ./inference_text



# model architecture
Arch:
  name: CLIP_vit_base_patch32_224_with_TextEncoder
  clip: "text"

Loss:
  Train:
    - ContrastiveLoss:
        margin: 1.0
        embedding_size: 512
        is_text_image_pairs: True
        weight: 1.0
  Eval:
    - ContrastiveLoss:
        margin: 1.0
        embedding_size: 512
        is_text_image_pairs: True
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
      name: Img2Dataset
      root_path: "./mscoco"
      split: "train"
      transform:        
        - ResizeImage:
            size: 224
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
      batch_size: 256
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

  Eval:
    dataset:
      name: Img2Dataset
      root_path: "./mscoco"
      split: "eval"
      transform:        
        - ResizeImage:
            size: 224
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
      batch_size: 1
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

```
用户可以根据自身需求，更改相应配置。注意arch参数请参照本文档。

**模型训练**。
```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train_multimodal.py \
        -c ./ppcls/configs/ram/CLIP.yaml
# 单卡
python3 tools/train_multimodal.py \
        -c ./ppcls/configs//ram/CLIP.yaml
```


<a name="3"></a>
## 3. CLIP动态、静态推理

**CLIP动态推理**。目前CLIP动态推理的输出结果为给定图片的embedding。
```bash
python3 tools/infer_multimodal.py \
    -c ./ppcls/configs/ram/CLIP.yaml \
    -o Global.pretrained_model="./output/ram/best_model"
```
**CLIP静态推理**。导出 inference model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/ram/CLIP.yaml \
    -o Global.pretrained_model="./output/CLIP/"
```
inference model 的路径默认在当前路径下 `./inference`
`./inference` 文件夹下应有如下文件结构：

```
├── inference
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```


切换到depoly目录下，并且使用deploy中的脚本进行推理前需要确认paddleclas为非本地安装, 如不是请进行切换，不然会出现包的导入错误。 

```shell
# 本地安装
pip install -e .
# 非本地安装
python setup.py install

# 进入deploy目录下
cd deploy
```

运行下面的命令，获取图像 `docs/images/inference_deployment/whl_demo.jpg` 的embedding。

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
<a name="4"></a>
## 4. CLIP落盘推理
CLIP落盘推理实现了将文本-图像嵌入后，使用向量数据库保存相关的嵌入向量。在查询时，可以直接通过对比数据库中，向量的余弦相似度实现文本查询图片，图片查询文本，图片查询图片。主要配置文件如下：

```yaml
Global:
  infer_imgs: "docs/images/inference_deployment/whl_demo.jpg"
  texts: "text_prompts.txt"
  inference_image_encoder_dir: "./inference_image"
  inference_text_encoder_dir: "./inference_text"
  batch_size: 1
  mode: "text-to-image"
  use_gpu: False
  embedding_size: 512
  enable_mkldnn: False
  cpu_num_threads: 10
  enable_benchmark: True
  use_fp16: False
  ir_optim: False # do not set it as True since there is a bug which leads the invaild initilize for predictor
  use_tensorrt: False
  gpu_mem: 8000
  enable_profile: False

PreProcess:
  transform_ops:
    - ResizeImage:
        size: 224
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.48145466, 0.4578275, 0.40821073]
        std: [0.26862954, 0.26130258, 0.27577711]
        order: ""
        channel_num: 3
    - ToCHWImage:


IndexProcess:
  index_method: "HNSW32" # supported: HNSW32, IVF, Flat
  image_index_dir: "./clip_image"
  text_index_dir: "./clip_text"
  index_operation: "new" # suported: "append", "remove", "new"
  delimiter: "\t"
  dist_type: "IP"
  embedding_size: 512
  batch_size: 1
  return_k: 1
  score_thres: 0.5

```
核心参数注释：
  - infer_imgs  模型参数，目前仅兼容vit为backbone的CLIP。
  - texts  输入类型参数，用于申明输入的类型。类型包括 "image"-图片以及"text"-文本。该参数仅作用于推理阶段。训练阶段此参数无实际意义。
  - inference_image_encoder_dir 图像编码器静态参数目录。
  - inference_text_encoder_dir 文本编码器静态参数目录。
  - batch_size 推理图片数量
  - mode 用于决定具体推理任务，具体包括: "image-to-text": 通过给定输入图像查询对应标签,"image-to-image"通过给定输入图像查询与输入图像相匹配的图像,"image_index_build"构建图像向量并且入库,"text_index_build"构建文本向量并且入库,"text-to-image"通过给定输入文本查询对应图像。
  - embedding_size 嵌入向量的维度，取决于CLIP嵌入向量的维度，默认512。
  - index_method 向量落库时的存储类型，如无特殊需求，始终默认为NHSW32。
  - image_index_dir 图像向量库所在路径。
  - text_index_dir 文本向量库所在路径。
  - return_k 返回top-k个相关的图像或文本。输出类型取决于mode设定的相关任务类型。
  - score_thres 相似度阈值。

**向量库构建**。向量库构建是使用CLIP实现下游任务的核心步骤。生成后的向量库路径为：1. 图像向量默认落库到image_index_dir指定的路径。2.文本向量默认落库到text_index_dir指定的路径。落库时，仅需将mode参数更改为"image_index_build"或"text_index_build"，使用如下shell命令生成相关向量库：

```shell
# linux使用`python3`，windows使用`python (-m)`来执行脚本
# 使用下面的命令使用 GPU 进行预测
python3 python/predictor_clip.py \
    -c deploy/configs/inference_clip.yaml \
# 使用下面的命令使用 CPU 进行预测
#更改 `config_infer.yaml` 配置文件后
python3 python/predictor_clip.py \
    -c deploy/configs/inference_clip.yaml \
```

**下游任务**。生成相关向量库后，即可以三种下游任务：1. 基于给定图像查询关联图像。2.基于给定图像查询符合图像描述的文本。3.基于给定的文本查询相关的图像。具体任务可以使用mode字段进行配置。配置后可以使用如下shell命令完成任务：
```shell
# linux使用`python3`，windows使用`python (-m)`来执行脚本
# 使用下面的命令使用 GPU 进行预测
python3 python/predictor_clip.py \
    -c deploy/configs/inference_clip.yaml \
# 使用下面的命令使用 CPU 进行预测
#更改 `config_infer.yaml` 配置文件后
python3 python/predictor_clip.py \
    -c deploy/configs/inference_clip.yaml \
```
任务1和3的输出示例为：
```python
[
  "docs/images/inference_deployment/whl_demo.jpg ",
]
```
任务2的输出示例为：
```python
[
  "a photo of chikien ",
]
```

**注意** 由于CLIP模型特性的约束，目前文本仅支持简单的prompts[https://github.com/openai/CLIP]文本.
<a name="5"></a>
## 5. 引用
```
@misc{beaumont-2021-img2dataset,
  author = {Romain Beaumont},
  title = {img2dataset: Easily turn large sets of image urls to an image dataset},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rom1504/img2dataset}}
}
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```