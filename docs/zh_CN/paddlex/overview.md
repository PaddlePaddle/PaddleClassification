
### 目录
- [1. 低代码全流程开发简介](#1-低代码全流程开发简介)
- [2. 图像分类和检索相关能力支持](#2-图像分类和检索相关能力支持)
- [3. 图像分类和检索相关模型产线列表和教程](#3-图像分类和检索相关模型产线列表和教程)
- [4. 图像分类和检索相关单功能模块列表和教程](#4-图像分类和检索相关单功能模块列表和教程)

<a name="1"></a>

## 1. 低代码全流程开发简介

飞桨低代码全流程开发工具[PaddleX](https://github.com/PaddlePaddle/PaddleX)，依托于PaddleClas的先进技术，支持了图像分类和检索领域的**低代码全流程**开发能力。通过低代码全流程开发，可实现简单且高效的模型使用、组合与定制。这将显著**减少模型开发的时间消耗**，**降低其开发难度**，大大加快模型在行业中的应用和推广速度。特色如下：

* 🎨 **模型丰富一键调用**：将通用图像分类、图像多标签分类、通用图像识别、人脸识别涉及的**98个模型**整合为6条模型产线，通过极简的**Python API一键调用**，快速体验模型效果。此外，同一套API，也支持目标检测、图像分割、文本图像智能分析、通用OCR、时序预测等共计**200+模型**，形成20+单功能模块，方便开发者进行**模型组合使用**。

* 🚀 **提高效率降低门槛**：提供基于**统一命令**和**图形界面**两种方式，实现模型简洁高效的使用、组合与定制。支持**高性能部署、服务化部署和端侧部署**等多种部署方式。此外，对于各种主流硬件如**英伟达GPU、昆仑芯、昇腾、寒武纪和海光**等，进行模型开发时，都可以**无缝切换**。

>**❗说明**：PaddleX 致力于实现产线级别的模型训练、推理与部署。模型产线是指一系列预定义好的、针对特定AI任务的开发流程，其中包含能够独立完成某类任务的单模型（单功能模块）组合。

<a name="2"></a>

## 2. 图像分类和检索相关能力支持

PaddleX中图像分类和图像检索的6条产线均支持本地**快速推理**，部分产线支持**在线体验**，您可以快速体验各个产线的预训练模型效果，如果您对产线的预训练模型效果满意，可以直接对产线进行[高性能推理](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/high_performance_inference.html)/[服务化部署](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/service_deploy.html)/[端侧部署](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/edge_deploy.html)，如果不满意，您也可以使用产线的**二次开发**能力，提升效果。完整的产线开发流程请参考[PaddleX产线使用概览](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/pipeline_develop_guide.html)或各产线使用教程。

此外，PaddleX为开发者提供了基于[云端图形化开发界面](https://aistudio.baidu.com/pipeline/mine)的全流程开发工具, 详细请参考[教程《零门槛开发产业级AI模型》](https://aistudio.baidu.com/practical/introduce/546656605663301)


<table >
    <tr>
        <td></td>
        <td>在线体验</td>
        <td>快速推理</td>
        <td>高性能部署</td>
        <td>服务化部署</td>
        <td>端侧部署</td>
        <td>二次开发</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
<tr>
        <td>通用图像分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td>图像多标签分类</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>通用图像识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>人脸识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
        <tr>
        <td>行人属性识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>车辆属性识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>

    
</table>

> ❗注：以上功能均基于GPU/CPU实现。PaddleX还可在昆仑、昇腾、寒武纪和海光等主流硬件上进行快速推理和二次开发。下表详细列出了模型产线的支持情况，具体支持的模型列表请参阅 [模型列表(NPU)](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_npu.html) // [模型列表(XPU)](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_xpu.html) // [模型列表(MLU)](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_mlu.html) // [模型列表DCU](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_dcu.html)。同时我们也在适配更多的模型，并在主流硬件上推动高性能和服务化部署的实施。


**🚀 国产化硬件能力支持**

<table>
  <tr>
    <th>产线名称</th>
    <th>昇腾 910B</th>
    <th>昆仑 R200/R300</th>
    <th>寒武纪 MLU370X8</th>
    <th>海光 Z100</th>
  </tr>
  <tr>
    <td>通用图像分类</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
</table>

<a name="3"></a>

## 3. 图像分类和检索相关模型产线列表和教程

- **通用图像分类产线**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_classification.html)
- **图像多标签分类产线**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html)
- **通用图像识别产线** [使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.html)
- **人脸识别产线** [使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/face_recognition.html)
- **行人属性识别产线** [使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute_recognition.html)
- **车辆属性识别产线** [使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute_recognition.html)

<a name="4"></a>

## 4. 图像分类和检索相关单功能模块列表和教程

- **图像分类模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_classification.html)
- **图像多标签分类模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_multilabel_classification.html)
- **图像特征模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_feature.html)
- **主体检测模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/mainbody_detection.html)
- **人脸检测模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/face_detection.html)
- **行人检测模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/human_detection.html)
- **行人属性识别模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.html)
- **车辆检测模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/vehicle_detection.html)
- **车辆属性识别模块**: [使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/vehicle_attribute_recognition.html)
