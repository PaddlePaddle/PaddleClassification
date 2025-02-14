# rfc_task-137_MoCo-v2模型PaddleClas实现设计文档）

|模型名称 | MoCov2模型 |
|---|---|
|相关paper| https://arxiv.org/pdf/2003.04297.pdf |
|参考项目| https://github.com/PaddlePaddle/PASSL https://github.com/facebookresearch/MoCo|
|提交作者 | 张乐 |
|提交时间 | 2022-03-11 |
|依赖飞桨版本 | PaddlePaddle2.4.1 |
|文件名 | rfc_task_137_model_MoCo-v2.md |

# MoCo-v2 模型PaddleClas实现设计文档
## 一、概述

MoCo-v2[<sup>2</sup>](#moco-v2)模型是在MoCo模型的基础上增加了数据增强、将单层fc替换为多层mlp、学习率衰减策略修改为consine衰减。因此，我们在此重点介绍MoCo模型。

MoCo[<sup>1</sup>](#moco-v1)模型本身是一个自监督对比学习框架，可以从大规模图像数据集中学习到良好的图像表示特征，其预训练模型可以无缝地嵌入许多视觉任务中，比如：图像分类、目标检测、分割等。

**MoCo框架简述**

**前向传播**

下面我们从输入$minibatchImgs=\{I_1,I_2,..I_N\}$ 数据的前向传播过程来简单讲解MoCo框架，首先对$I_n$分别进行变换$view_1$和$view_2$：
$$I^{view1}_n=view_1(I_n)$$
$$I^{view2}_n=view_2(I_n)$$
其中，$view_1$和$view_2$表示一系列图像预处理变换（随机裁切、灰度化、均值化等，具体详见paper Source Code），minibatch大小为$N$。这样每幅输入图像$I_n$就会得到两个变换图像$I^{view1}_n$和$I^{view2}_n$。

接着将$I^{view1}_n$和$I^{view2}_n$分别送入两个编码器，则：
$$q_n=L2_{normalization}(Encoder_1(I^{view1}_n))$$
$$k_n=L2_{normalization}(Encoder_2(I^{view2}_n))$$

其中$q_n$和$k_n$的特征维度均为k,  $Encoder_1$和$Encoder_2$分别是ResNet50的backbone网络串联一个MLP网络组成。

为了满足对比学习任务的条件,需要正负样本来进行学习。作者自然而然将输入的样本都看作正样本，至于负样本，则通过构建一个**动态**$Dict_{K\times C}$维度的超大字典，通过将正样本集合$q_+=\{q_1,q_2...q_N\}$和$k_+=\{k_1,k_2...k_N\}$一一做向量点乘求和相加来计算$Loss_+$：

$$Loss_+=\{l^{1}_+;l^{2}_+; ...;l^{N}_+\}=\{ q_1\cdot k_1; q_2\cdot k_2;...; q_n\cdot k_n \}; Loss_+\in N \times 1$$


$Loss_-$的计算过程为：
$$l^{n,k}_-=q_n \cdot Dict_{:,n};Loss_-\in N \times C$$


最后的loss为：
$$Loss=concat(Loss_+, Loss_-)\in N \times (1+C)$$
可以看到字典$Dict$在整个图像表示的学习过程中可以看作一个隐特征空间，作者发现，该字典设置的越大，视觉表示学习的效果就越好。其中，每次在做完前向传播后，需要将当前的minibatch以**队列入队**的形式将$k_n$加入到字典$Dict$中,并同时将最旧时刻的minibatch**出队**。

学习的目标函数采用交叉熵损失函数如下所示：

$$Loss_{crossentropy}=-log \cdot \frac{exp(l_+/ \tau)}{ \sum exp(l_n / \tau)}$$

其中超参数$\tau$取0.07

**反向梯度传播**

在梯度反向传播过程中，梯度传播只用来更新$Encoder_1$的参数$Param_{Encoder_1}$,为了不影响动态词典$Dict$的视觉表示特征一致性，$Encoder_2$的参数$Param_{Encoder_1}$更新过程为：

$$Param_{Encoder_2}=m \cdot Param_{Encoder_2} + ( 1- m ) \cdot  Param_{Encoder_1} $$
其中，超参数$m$取0.999

## 二、设计思路与实现方案

### 模型backbone（PaddleClas已有实现）

- ResNet50的backbone(去除最后的全连接层)
- MLP由 两个全连接层FC1 $ 2048 \times 2048 $ 和FC2 $ 2048 \times 128 $ 构成
- 动态字典大小为$65536$
### optimizer
- SGD:随机梯度下降优化器
- 初始学习率 $0.03$
- 权重衰减：$1e-4$
- momentum of SGD： $0.9$

### 训练策略（PaddleClas已有实现）
- batch-size：256
- 单机8块V100
- 在每个GPU上做shuffle_BN
- 共迭代$epochs:200$

- lr schedule 在$epch=[120, 160]$, $lr=lr*.0.1$
- 学习率衰减策略$cosine $

### metric（PaddleClas已有实现）
- top1
- top5

### dataset
- 数据集：ImageNet
- 数据增强（PaddleClas已有基本变换实现）
```Python
   #pytorch 代码
   augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

 ```
- 图像对随机变换和高斯模糊（**PSSL已有基本变换实现,需要转为PaddleClas项目实现**）

 ```python
# pytorch 代码
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
 ```

### PSSL项目和PaddleClas项目框架对比

- 两个项目基础模型ResNet50的每层参数名称不同，需要将PASSL项目的训练权重转化为PaddleClas项目使用
- PSSL项目采用Register类方式将模型的architecture、backbone、neck、head、数据集、优化器、钩子函数链接在一起，使得整个模型的训练过程都可以通过命令行提供一份yaml文件搞定，这一点与PaddleClas项目类似

## 三、功能模块测试方法
|功能模块|测试方法|
|---|---|
|前向完全对齐|给定相同的输入，分别对比PaddleClas实现的模型输出是否和官方的Pytorch版本相同|
|反向完全对齐|给定相同的输入检查反向参数更新，分别对比PaddleClas实现和官方的Pytorch版本参数更新是否一致|
|图像预处理|对照官方实现，编写paddle版本|
|超参数配置|保持和官方实现一致|
|训练环境|最好也是8块V100显卡环境，采用单机多卡分布式训练方式，和官方保持一致|
|精度对齐|在提供的小数据集上预训练并finetune后，实现精度和原PSSL项目模型相同|

## 四、可行性分析和排期规划
|时间|开发排期规划|时长|
|---|---|---|
|03.11-03.19|熟悉相关工具、前向对齐|9days|
|03.20-04.02|反向对齐|14days|
|04.03-04.16|训练对齐|14days|
|04.16-04.29|代码合入|14days|

## 五、风险点与影响面

风险点:
- MoCo模型训练后一般作为图像特征提取器使用，并不存在所谓的推理过程
- **PaddleClas中所有模型和算法需要通过飞桨训推一体认证,当前只需要通过新增模型只需要通过训练和推理的基础认证即可**。但是这个与MoCo模型的训练推理原则相违背，是否可以对MoCo-v2模型的认证给出明确的指定
- 合入代码题目是MoCo-v2,代码合入的时候是否需要同时考虑MoCo-v1代码模块（原PSSL项目有该项实现）
- 原PSSL有MoCo-Clas分类模型，代码合入的时候是否需要同时加入此模块（原PSSL项目有该项实现）

影响面：
数据的Dataloader、数据增强和model均为新增脚本，不对其它模块构成影响

# 名词解释
MoCo(Momentum Contrast，动量对比)
# 附件及参考资料
<div id="moco-v1"></div>
  [1] He K, Fan H, Wu Y, et al. Momentum contrast for unsupervised visual representation learning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 9729-9738.

<div id="moco-v2"></div>
  [2] Chen X, Fan H, Girshick R, et al. Improved baselines with momentum contrastive learning[J]. arXiv preprint arXiv:2003.04297, 2020.
