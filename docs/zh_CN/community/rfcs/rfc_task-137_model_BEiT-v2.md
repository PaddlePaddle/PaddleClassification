# rfc_task-137_model_BEiT-v2模型PaddleClas实现设计文档）

|模型名称 | BEiTv2 |
|---|---|
|相关paper| https://arxiv.org/pdf/2208.06366.pdf |
|参考项目|PaddleClas比赛导师提供|
|提交作者 | 刘学文 |
|提交时间 | 2022-03-11 |
|依赖飞桨版本 | PaddlePaddle2.4.1 |
|文件名 | rfc_task_137_model_BEiT-v2.md |
----------------------------------------------------------------

# BEiTv2模型PaddleClas实现设计文档
## 一、概述
BEiT-v2[<sup>1</sup>](#BEiT-v2)模型是BEiT[<sup>2</sup>](#BEiT-v1)模型的V2版本。BEiT是一个两阶段的算法：首先通过一个dVAE将图像映射成离散的视觉标志（Visual Token），然后再通过视觉Transformer学习带掩码的图像Patch到视觉标志的映射。

BEiT-v2的提出，是为了解决BEiT未对第一阶段中dVAE学习到的语义空间进行深入的探讨和优化的问题。它的核心是通过训练好的模型作为Teacher来指导视觉标志的学习，同时，引入了标志符来学习整个图像的特征，以提高准确率。

**算法详解**

BEiT-v2是一个两阶段的模型

**第一阶段**

第一阶段是VQ-KD的训练，VQ-KD由两部分组成，分别是*Tokenizer*以及*Decoder*

VQ-KD的作用是将输入图像转化为视觉标志，即将输入图像 $x$ 转化未视觉标志 $\boldsymbol{z}=\left[z_{1}, \cdots, z_{N}\right] \in \mathcal{V}^{(H / P) \times(W / P)}$ , 其中 $\mathcal{V}$ 指的是视觉字典。

**Tokenizer**

Tokenizer的计算分成两步：它首先使用ViT将输入图像编码成特征向量，然后使用从码本中找最近邻。
假设图像序列 $\left\{\boldsymbol{x}_{i}^{p}\right\}_{i=1}^{N}$ 编码成的序列表示为 $\left\{h_{i}^{p}\right\}_{i=1}^{N}$，码本的嵌入表示为 $\left\{\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_K\right\}$ ，那么对于第 $i$ 个图像的patch对应的视觉标志可以通过下面的式子确定：
$$z_i=\arg \min _j\left\|\ell_2\left(\boldsymbol{h}_i\right)-\ell_2\left(\boldsymbol{e}_j\right)\right\|_2$$
其中 $\ell_2$ 是特征的L2正则化

**Decoder**

Decoder也是一个多层的Transformer，当Tokenizer将图像表示到视觉标志后，通过将这些视觉标志正则化，可以将它输入到解码器中，得到最终的输出 $\left\{\boldsymbol{o}_i\right\}_{i=1}^N$ 。输出向量的目的是重建训练好的模型作为Teacher指导的特征。

**损失函数**

VQ-KD的损失函数可以表示最大化模型输出以及Teacher生成的特征相似度并最小化生成特征和视觉单词的距离。因为存在不可导操作，所以损失函数的内容如下式
$$\max \sum_{x \in \mathcal{D}} \sum_{i=1}^N \cos \left(\boldsymbol{o}_i, \boldsymbol{t}_i\right)-\left\|\operatorname{sg}\left[\ell_2\left(\boldsymbol{h}_i\right)\right]-\ell_2\left(\boldsymbol{v}_{z_i}\right)\right\|_2^2-\left\|\ell_2\left(\boldsymbol{h}_i\right)-\operatorname{sg}\left[\ell_2\left(\boldsymbol{v}_{z_i}\right)\right]\right\|_2^2$$
其中 $\operatorname{sg}[.]$ 表示停止梯度计算操作， $\mathcal{D}$ 表示用来Tokenizer的训练数据集。

**第二阶段**

第二阶段是预训练。预训练分为两个部分：掩码图像模型的训练和标志的训练。

**掩码图像模型**

BEiT-v2的预训练遵循了和BEiTv1类似的方式（MIM），不同的是它在输入数据中拼接了[CLS]标志。[CLS]标志与BEiTv1中得到的 $\boldsymbol{x}_i^{\mathcal{M}}$ 共同输入到视觉Transformer中，通过视觉Transformer的计算，可以得到模型的输出 $\left\{\boldsymbol{h}_i\right\}_{i=0}^N$ 。

在视觉Transformer之后添加一个MIM的输出头，用于预测图像patch对应的视觉标志，对于每个输出，使用softmax损失函数预测每个patch的输出概率，表示为下式：
$$p\left(\boldsymbol{z}_i \mid \boldsymbol{h}_i\right)=\operatorname{softmax}_{\boldsymbol{z}_i}\left(\boldsymbol{W}_c \boldsymbol{h}_i+\boldsymbol{b}_c\right)$$
其中 $\boldsymbol{W}$ 和 $\boldsymbol{b}$ 分别是权值矩阵和偏置向量。最终MIM的损失函数表示为：
$$\mathcal{L}_{\mathrm{MIM}}=-\sum_{x \in \mathcal{D}} \sum_{i \in \mathcal{M}} \log p\left(z_i \mid x^{\mathcal{M}}\right)$$
其中 $\boldsymbol{z}_i$ 是原始图像的视觉标志， $\mathcal{D}$ 表示与训练图像。

**预训练全局表示**

对[CLS]标志进行训练来捕获图像的全局信息，[CLS]的预训练的输入是由第 $l$ 层视觉Transformer的特征向量和第 $L$ 层[CLS]的特征向量拼接而成（具体可以见论文[<sup>1</sup>](#BEiT-v2)图三虚线框内），接下来将特征 $S$ 输入到一个两层的Transformer中来预测掩码图像的视觉标志.



## 二、设计思路与实现方案

### backbone
- Transformer

### optimizer(PaddleClas已有实现)
- adamw
- adam
- 初始学习率： $5e-4$
- 权重衰减：$0.05$
- momentum ： $0.9$

### loss
- 见算法详解

### dataset
- ImageNet-1K
- ADE20K

### metric
#### image classification
- top-1 accuracy
#### semantic segmentation
- mIoU

### 训练策略
- 16块V100 32GB英伟达显卡
#### image classification
- 训练和评估都是在ImageNet-1K上
##### Fine-tuning setup
- 遵循BEiT中提出的方法微调BEiT-v2模型
##### Linear probing
- 将表征层的特征固定，只通过监督数据去训练分类器
#### Semantic Segmentation
- 实验在ADE20K上进行
- 使用UperNet[<sup>3</sup>](#UperNet)任务层，在输入分辨率为512×512的情况下，对模型进行16万次迭代微调。


## 三、功能模块测试方法
|功能模块|测试方法|
|---|---|
|前向完全对齐|给定相同的输入，分别对比PaddleClas实现的模型输出是否和官方的Pytorch版本相同|
|反向完全对齐|给定相同的输入检查反向参数更新，分别对比PaddleClas实现和官方的Pytorch版本参数更新是否一致|
|图像预处理|对照官方实现，编写paddle版本|
|超参数配置|保持和官方实现一致|
|训练环境|最好也是16块V100显卡环境，采用单机多卡分布式训练方式，和官方保持一致|
|精度对齐|在提供的小数据集上预训练并finetune后，实现精度和论文对齐|
## 四、可行性分析和排期规划
|时间|开发排期规划|时长|
|---|---|---|
|03.11-03.19|熟悉相关工具、前向对齐|9days|
|03.20-03.26|反向对齐|7days|
|03.27-04.09|训练对齐|14days|
|04.10-04.16|代码合入|7days|

## 五、风险点与影响面
风险点：
- BEiT-v2模型分成多个预训练模型，合入PaddleClas代码量较多

# 名词解释
MIM(Masked Image Model)
# 附件及参考资料
<div id="BEiT-v2"></div>
  [1] Peng Z, Dong L, Bao H, et al. Beit v2: Masked image modeling with vector-quantized visual tokenizers[J]. arXiv preprint arXiv:2208.06366, 2022.

<div id="BEiT-v1"></div>
  [2] Bao H, Dong L, Piao S, et al. Beit: Bert pre-training of image transformers[J]. arXiv preprint arXiv:2106.08254, 2021.

<div id="UperNet"></div>
  [3] Xiao T, Liu Y, Zhou B, et al. Unified perceptual parsing for scene understanding[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 418-434.
