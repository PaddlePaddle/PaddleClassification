# 更新日志

- **🔥2024.10.1 发布PaddleClas release/2.6**:
  *  飞桨一站式全流程开发工具PaddleX，依托于PaddleClas的先进技术，支持了图像分类和图像检索领域的**一站式全流程**开发能力：
     * 🎨 [**模型丰富一键调用**](docs/zh_CN/paddlex/quick_start.md)：将通用图像分类、图像多标签分类、通用图像识别、人脸识别涉及的**98个模型**整合为6条模型产线，通过极简的**Python API一键调用**，快速体验模型效果。此外，同一套API，也支持目标检测、图像分割、文本图像智能分析、通用OCR、时序预测等共计**200+模型**，形成20+单功能模块，方便开发者进行**模型组合使用**。
     * 🚀 [**提高效率降低门槛**](docs/zh_CN/paddlex/overview.md)：提供基于**统一命令**和**图形界面**两种方式，实现模型简洁高效的使用、组合与定制。支持**高性能部署、服务化部署和端侧部署**等多种部署方式。此外，对于各种主流硬件如**英伟达GPU、昆仑芯、昇腾、寒武纪和海光**等，进行模型开发时，都可以**无缝切换**。
  * 新增图像分类算法[**MobileNetV4、StarNet、FasterNet**](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/cv_modules/image_classification.md)

- 2022.9.13 发布超轻量图像识别系统[PP-ShiTuV2](docs/zh_CN/models/PP-ShiTu/README.md)：
  - recall1精度提升8个点，覆盖商品识别、垃圾分类、航拍场景等[20+识别场景](docs/zh_CN/deployment/PP-ShiTu/application_scenarios.md)，
  - 新增[库管理工具](./deploy/shitu_index_manager/)，[Android Demo](./docs/zh_CN/quick_start/quick_start_recognition.md)全新体验。

- 2022.9.4 新增[生鲜产品自主结算范例库](./docs/zh_CN/samples/Fresh_Food_Recogniiton/README.md)，具体内容可以在AI Studio上体验。
- 2022.6.15 发布[PULC超轻量图像分类实用方案](docs/zh_CN/training/PULC.md)，CPU推理3ms，精度比肩SwinTransformer，覆盖人、车、OCR场景九大常见任务。
- 2022.5.23 新增[人员出入管理范例库](https://aistudio.baidu.com/aistudio/projectdetail/4094475)，具体内容可以在 AI Studio 上体验。
- 2022.5.20 上线[PP-HGNet](./docs/zh_CN/models/ImageNet1k/PP-HGNet.md), [PP-LCNetv2](./docs/zh_CN/models/ImageNet1k/PP-LCNetV2.md)。
- 2022.4.21 新增 CVPR2022 oral论文 [MixFormer](https://arxiv.org/pdf/2204.02557.pdf) 相关[代码](https://github.com/PaddlePaddle/PaddleClas/pull/1820/files)。
- 2021.11.1 发布[PP-ShiTu技术报告](https://arxiv.org/pdf/2111.00775.pdf)，新增饮料识别demo。
- 2021.10.23 发布轻量级图像识别系统PP-ShiTu，CPU上0.2s即可完成在10w+库的图像识别。[点击这里](quick_start/quick_start_recognition.md)立即体验。
- 2021.09.17 发布PP-LCNet系列超轻量骨干网络模型, 在Intel CPU上，单张图像预测速度约5ms，ImageNet-1K数据集上Top1识别准确率达到80.82%，超越ResNet152的模型效果。PP-LCNet的介绍可以参考[论文](https://arxiv.org/pdf/2109.15099.pdf), 或者[PP-LCNet模型介绍](../models/PP-LCNet.md)，相关指标和预训练权重可以从 [这里](models/ImageNet1k/model_list.md)下载。
- 2021.08.11 更新 7 个[FAQ](FAQ/faq_2021_s2.md)。
- 2021.06.29 添加 Swin-transformer 系列模型，ImageNet1k 数据集上 Top1 acc 最高精度可达 87.2%；支持训练预测评估与 whl 包部署，预训练模型可以从[这里](models/ImageNet1k/model_list.md)下载。
- 2021.06.22,23,24 PaddleClas 官方研发团队带来技术深入解读三日直播课。课程回放：[https://aistudio.baidu.com/aistudio/course/introduce/24519](https://aistudio.baidu.com/aistudio/course/introduce/24519)
- 2021.06.16 PaddleClas v2.2 版本升级，集成 Metric learning，向量检索等组件。新增商品识别、动漫人物识别、车辆识别和 logo 识别等 4 个图像识别应用。新增 LeViT、Twins、TNT、DLA、HarDNet、RedNet 系列 30 个预训练模型。
- 2021.04.15
   - 添加 `MixNet_L` 和 `ReXNet_3_0` 系列模型，在 ImageNet-1k 上 `MixNet` 模型 Top1 Acc 可达 78.6%，`ReXNet` 模型可达 82.09%
- 2021.01.27
   * 添加 ViT 与 DeiT 模型，在 ImageNet 上，ViT 模型 Top-1 Acc 可达 81.05%，DeiT 模型可达 85.5%。
- 2021.01.08
    * 添加 whl 包及其使用说明，直接安装 paddleclas whl 包，即可快速完成模型预测。
- 2020.12.16
    * 添加对 cpp 预测的 tensorRT 支持，预测加速更明显。
- 2020.12.06
    * 添加 SE_HRNet_W64_C_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.8475。
- 2020.11.23
    * 添加 GhostNet_x1_3_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.7938。
- 2020.11.09
    * 添加 InceptionV3 结构和模型，在 ImageNet 上 Top-1 Acc 可达 0.791。
- 2020.10.20
    * 添加 Res2Net50_vd_26w_4s_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.831；添加 Res2Net101_vd_26w_4s_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.839。
- 2020.10.12
    * 添加 Paddle-Lite demo。
- 2020.10.10
    * 添加 cpp inference demo。
    * 添加 FAQ 30 问。
- 2020.09.17
    * 添加 HRNet_W48_C_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.836；添加 ResNet34_vd_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.797。

* 2020.09.07
    * 添加 HRNet_W18_C_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.81162；添加 MobileNetV3_small_x0_35_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.5555。

* 2020.07.14
    * 添加 Res2Net200_vd_26w_4s_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 85.13%。
    * 添加 Fix_ResNet50_vd_ssld_v2 模型，，在 ImageNet 上 Top-1 Acc 可达 84.0%。

* 2020.06.17
    * 添加英文文档。

* 2020.06.12
    * 添加对 windows 和 CPU 环境的训练与评估支持。

* 2020.05.17
    * 添加混合精度训练。

* 2020.05.09
    * 添加 Paddle Serving 使用文档。
    * 添加 Paddle-Lite 使用文档。
    * 添加 T4 GPU 的 FP32/FP16 预测速度 benchmark。

* 2020.04.10:
    * 第一次提交。
