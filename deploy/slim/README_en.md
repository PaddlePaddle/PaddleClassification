
## Introduction to Slim

Generally, a more complex model would achive better performance in the task, but it also leads to some redundancy in the model.  This part provides the function of compressing the model, including two parts: model quantization (offline quantization training and online quantization training) and model pruning.
Quantization is a technique that reduces this redundancy by reducing the full precision data to a fixed number, so as to reduce model calculation complexity and improve model inference performance.

Model pruning cuts off the unimportant convolution kernel in CNN to reduce  the amount of model parameters, so as to reduce the computational  complexity of the model.

It is recommended that you could understand following pages before reading this example：
- [The training strategy of PaddleClas models](../../docs/en/tutorials/getting_started_en.md)
- [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

## Quick Start
 After training a model, if you want to further compress the model size and  speed up the prediction, you can use quantization or pruning to compress the model according to the following steps.

1. Install PaddleSlim
2. Prepare trained model
3. Model compression
4. Export inference model
5. Deploy quantization inference model


### 1. Install PaddleSlim

* Install by pip.

```bash
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* Install from source code to get the lastest features.

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python setup.py install
```


### 2. Download Pretrain Model
PaddleClas provides a series of trained [models](../../docs/en/models/models_intro_en.md).
If the model to be quantified is not in the list, you need to follow the [Regular Training](../../docs/en/tutorials/getting_started_en.md) method to get the trained model.

### 3. Model Compression

Go to the root directory of PaddleClas

```bash
cd PaddleClase
```

The code of slim is located in `deploy/slim`

#### 3.1 Model Quantization

Quantization training includes offline quantization  and online quantization training.

##### 3.1.1 Online quantization training

Online quantization training is more effective. It is necessary to load the pre-trained model.
After the quantization strategy is defined, the model can be quantified.

The training command is as follow:

* CPU/Single GPU

If using GPU, change the `cpu` to `gpu` in the following command.

```bash
python3.7 tools/train.py -c ppcls/configs/slim/ResNet50_vd_quantization.yaml -o Global.device=cpu
```

The description of `yaml` file can be found  in this [doc](../../docs/en/tutorials/config_en.md). To get better accuracy, the `pretrained model`is used in `yaml`.


* Distributed training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
      tools/train.py \
      -m train \
      -c ppcls/configs/slim/ResNet50_vd_quantization.yaml
```

##### 3.1.2 Offline quantization

**Attention**:  At present, offline quantization must use `inference model` as input, which can be exported by trained model.  The process of exporting `inference model` for trained model can refer to this [doc](../../docs/en/inference.md).

Generally speaking, the offline quantization gets more loss of accuracy than online qutization training.

After getting `inference model`, we can run following command to get offline quantization model.

```
python3.7 deploy/slim/quant_post_static.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

`Global.save_inference_dir` is the directory storing the `inference model`.

If run successfully, the directory `quant_post_static_model` is generated in `Global.save_inference_dir`, which stores the offline quantization model that can be used for deploy directly.

#### 3.2 Model Pruning

- CPU/Single GPU

If using GPU, change the `cpu` to `gpu` in the following command.

```bash
python3.7 tools/train.py -c ppcls/configs/slim/ResNet50_vd_prune.yaml -o Global.device=cpu
```

- Distributed training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
      tools/train.py \
      -c ppcls/configs/slim/ResNet50_vd_prune.yaml
```



### 4. Export inference model

After getting the compressed model, we can export it as inference model for predictive deployment. Using pruned model as example:

```bash
python3.7 tools/export.py \
    -c ppcls/configs/slim/ResNet50_vd_prune.yaml \
    -o Global.pretrained_model=./output/ResNet50_vd/best_model
    -o Global.save_inference_dir=./inference
```

### 5. Deploy
The derived model can be converted through the `opt tool` of PaddleLite.

For compresed model deployment, please refer to [Mobile terminal model deployment](../lite/readme_en.md)

## Notes:

* In quantitative training, it is suggested to load the pre-trained model obtained from conventional training to accelerate the convergence of quantitative training.
* In quantitative training, it is suggested that the initial learning rate should be changed to `1 / 20 ~ 1 / 10` of the conventional training, and the training epoch number should be changed to `1 / 5 ~ 1 / 2` of the conventional training. In terms of learning rate strategy, it's better to train with warmup, other configuration information is not recommended to be changed.
