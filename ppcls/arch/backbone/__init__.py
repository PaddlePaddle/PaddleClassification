# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import inspect

from .legendary_models.mobilenet_v1 import MobileNetV1_x0_25, MobileNetV1_x0_5, MobileNetV1_x0_75, MobileNetV1
from .legendary_models.mobilenet_v3 import MobileNetV3_small_x0_35, MobileNetV3_small_x0_5, MobileNetV3_small_x0_75, MobileNetV3_small_x1_0, MobileNetV3_small_x1_25, MobileNetV3_large_x0_35, MobileNetV3_large_x0_5, MobileNetV3_large_x0_75, MobileNetV3_large_x1_0, MobileNetV3_large_x1_25
from .legendary_models.mobilenet_v4 import MobileNetV4_conv_small, MobileNetV4_conv_medium, MobileNetV4_conv_large, MobileNetV4_hybrid_medium, MobileNetV4_hybrid_large
from .model_zoo.fasternet import FasterNet_T0, FasterNet_T1, FasterNet_T2, FasterNet_S, FasterNet_M, FasterNet_L
from .model_zoo.starnet import StarNet_S1, StarNet_S2, StarNet_S3, StarNet_S4
from .legendary_models.resnet import ResNet18, ResNet18_vd, ResNet34, ResNet34_vd, ResNet50, ResNet50_vd, ResNet101, ResNet101_vd, ResNet152, ResNet152_vd, ResNet200_vd
from .legendary_models.vgg import VGG11, VGG13, VGG16, VGG19
from .legendary_models.inception_v3 import InceptionV3
from .legendary_models.hrnet import HRNet_W18_C, HRNet_W30_C, HRNet_W32_C, HRNet_W40_C, HRNet_W44_C, HRNet_W48_C, HRNet_W60_C, HRNet_W64_C, SE_HRNet_W64_C
from .legendary_models.pp_lcnet import PPLCNetBaseNet, PPLCNet_x0_25, PPLCNet_x0_35, PPLCNet_x0_5, PPLCNet_x0_75, PPLCNet_x1_0, PPLCNet_x1_5, PPLCNet_x2_0, PPLCNet_x2_5
from .legendary_models.pp_lcnet_v2 import PPLCNetV2_small, PPLCNetV2_base, PPLCNetV2_large
from .legendary_models.esnet import ESNet_x0_25, ESNet_x0_5, ESNet_x0_75, ESNet_x1_0
from .legendary_models.pp_hgnet import PPHGNet_tiny, PPHGNet_small, PPHGNet_base
from .legendary_models.pp_hgnet_v2 import PPHGNetV2_B0, PPHGNetV2_B1, PPHGNetV2_B2, PPHGNetV2_B3, PPHGNetV2_B4, PPHGNetV2_B5, PPHGNetV2_B6

from .model_zoo.resnet_vc import ResNet50_vc
from .model_zoo.resnext import ResNeXt50_32x4d, ResNeXt50_64x4d, ResNeXt101_32x4d, ResNeXt101_64x4d, ResNeXt152_32x4d, ResNeXt152_64x4d
from .model_zoo.resnext_vd import ResNeXt50_vd_32x4d, ResNeXt50_vd_64x4d, ResNeXt101_vd_32x4d, ResNeXt101_vd_64x4d, ResNeXt152_vd_32x4d, ResNeXt152_vd_64x4d
from .model_zoo.res2net import Res2Net50_26w_4s, Res2Net50_14w_8s
from .model_zoo.res2net_vd import Res2Net50_vd_26w_4s, Res2Net101_vd_26w_4s, Res2Net200_vd_26w_4s
from .model_zoo.se_resnet_vd import SE_ResNet18_vd, SE_ResNet34_vd, SE_ResNet50_vd
from .model_zoo.se_resnext_vd import SE_ResNeXt50_vd_32x4d, SE_ResNeXt50_vd_32x4d, SENet154_vd
from .model_zoo.se_resnext import SE_ResNeXt50_32x4d, SE_ResNeXt101_32x4d, SE_ResNeXt152_64x4d
from .model_zoo.dpn import DPN68, DPN92, DPN98, DPN107, DPN131
from .model_zoo.dsnet import DSNet_tiny, DSNet_small, DSNet_base
from .model_zoo.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264
from .model_zoo.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetB0_small
from .model_zoo.efficientnet_v2 import EfficientNetV2_S
from .model_zoo.resnest import ResNeSt50_fast_1s1x64d, ResNeSt50, ResNeSt101, ResNeSt200, ResNeSt269
from .model_zoo.googlenet import GoogLeNet
from .model_zoo.mobilenet_v2 import MobileNetV2_x0_25, MobileNetV2_x0_5, MobileNetV2_x0_75, MobileNetV2, MobileNetV2_x1_5, MobileNetV2_x2_0
from .model_zoo.mobilefacenet import MobileFaceNet
from .model_zoo.shufflenet_v2 import ShuffleNetV2_x0_25, ShuffleNetV2_x0_33, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0, ShuffleNetV2_swish
from .model_zoo.ghostnet import GhostNet_x0_5, GhostNet_x1_0, GhostNet_x1_3
from .model_zoo.alexnet import AlexNet
from .model_zoo.inception_v4 import InceptionV4
from .model_zoo.xception import Xception41, Xception65, Xception71
from .model_zoo.xception_deeplab import Xception41_deeplab, Xception65_deeplab
from .model_zoo.resnext101_wsl import ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl
from .model_zoo.squeezenet import SqueezeNet1_0, SqueezeNet1_1
from .model_zoo.darknet import DarkNet53
from .model_zoo.regnet import RegNetX_200MF, RegNetX_400MF, RegNetX_600MF, RegNetX_800MF, RegNetX_1600MF, RegNetX_3200MF, RegNetX_4GF, RegNetX_6400MF, RegNetX_8GF, RegNetX_12GF, RegNetX_16GF, RegNetX_32GF
from .model_zoo.vision_transformer import ViT_small_patch16_224, ViT_base_patch16_224, ViT_base_patch16_384, ViT_base_patch32_384, ViT_large_patch16_224, ViT_large_patch16_384, ViT_large_patch32_384
from .model_zoo.distilled_vision_transformer import DeiT_tiny_patch16_224, DeiT_small_patch16_224, DeiT_base_patch16_224, DeiT_tiny_distilled_patch16_224, DeiT_small_distilled_patch16_224, DeiT_base_distilled_patch16_224, DeiT_base_patch16_384, DeiT_base_distilled_patch16_384
from .legendary_models.swin_transformer import SwinTransformer_tiny_patch4_window7_224, SwinTransformer_small_patch4_window7_224, SwinTransformer_base_patch4_window7_224, SwinTransformer_base_patch4_window12_384, SwinTransformer_large_patch4_window7_224, SwinTransformer_large_patch4_window12_384
from .model_zoo.swin_transformer_v2 import SwinTransformerV2_tiny_patch4_window8_256, SwinTransformerV2_small_patch4_window8_256, SwinTransformerV2_base_patch4_window8_256, SwinTransformerV2_tiny_patch4_window16_256, SwinTransformerV2_small_patch4_window16_256, SwinTransformerV2_base_patch4_window16_256, SwinTransformerV2_base_patch4_window24_384, SwinTransformerV2_large_patch4_window16_256, SwinTransformerV2_large_patch4_window24_384
from .model_zoo.cswin_transformer import CSWinTransformer_tiny_224, CSWinTransformer_small_224, CSWinTransformer_base_224, CSWinTransformer_large_224, CSWinTransformer_base_384, CSWinTransformer_large_384
from .model_zoo.mixnet import MixNet_S, MixNet_M, MixNet_L
from .model_zoo.rexnet import ReXNet_1_0, ReXNet_1_3, ReXNet_1_5, ReXNet_2_0, ReXNet_3_0
from .model_zoo.twins import pcpvt_small, pcpvt_base, pcpvt_large, alt_gvt_small, alt_gvt_base, alt_gvt_large
from .model_zoo.levit import LeViT_128S, LeViT_128, LeViT_192, LeViT_256, LeViT_384
from .model_zoo.dla import DLA34, DLA46_c, DLA46x_c, DLA60, DLA60x, DLA60x_c, DLA102, DLA102x, DLA102x2, DLA169
from .model_zoo.rednet import RedNet26, RedNet38, RedNet50, RedNet101, RedNet152
from .model_zoo.tnt import TNT_small, TNT_base
from .model_zoo.hardnet import HarDNet68, HarDNet85, HarDNet39_ds, HarDNet68_ds
from .model_zoo.cspnet import CSPDarkNet53
from .model_zoo.pvt_v2 import PVT_V2_B0, PVT_V2_B1, PVT_V2_B2_Linear, PVT_V2_B2, PVT_V2_B3, PVT_V2_B4, PVT_V2_B5
from .model_zoo.mobilevit import MobileViT_XXS, MobileViT_XS, MobileViT_S
from .model_zoo.repvgg import RepVGG_A0, RepVGG_A1, RepVGG_A2, RepVGG_B0, RepVGG_B1, RepVGG_B2, RepVGG_B1g2, RepVGG_B1g4, RepVGG_B2g4, RepVGG_B3, RepVGG_B3g4, RepVGG_D2se
from .model_zoo.van import VAN_B0, VAN_B1, VAN_B2, VAN_B3
from .model_zoo.peleenet import PeleeNet
from .model_zoo.foundation_vit import CLIP_vit_base_patch32_224, CLIP_vit_base_patch16_224, CLIP_vit_large_patch14_336, CLIP_vit_large_patch14_224, BEiTv2_vit_base_patch16_224, BEiTv2_vit_large_patch16_224, CAE_vit_base_patch16_224, EVA_vit_giant_patch14, MOCOV3_vit_small, MOCOV3_vit_base, MAE_vit_huge_patch14, MAE_vit_large_patch16, MAE_vit_base_patch16
from .model_zoo.convnext import ConvNeXt_tiny, ConvNeXt_small, ConvNeXt_base_224, ConvNeXt_base_384, ConvNeXt_large_224, ConvNeXt_large_384
from .model_zoo.nextvit import NextViT_small_224, NextViT_base_224, NextViT_large_224, NextViT_small_384, NextViT_base_384, NextViT_large_384
from .model_zoo.cae import cae_base_patch16_224, cae_large_patch16_224
from .model_zoo.cvt import CvT_13_224, CvT_13_384, CvT_21_224, CvT_21_384, CvT_W24_384
from .model_zoo.micronet import MicroNet_M0, MicroNet_M1, MicroNet_M2, MicroNet_M3
from .model_zoo.mobilenext import MobileNeXt_x0_35, MobileNeXt_x0_5, MobileNeXt_x0_75, MobileNeXt_x1_0, MobileNeXt_x1_4
from .model_zoo.mobilevit_v2 import MobileViTV2_x0_5, MobileViTV2_x0_75, MobileViTV2_x1_0, MobileViTV2_x1_25, MobileViTV2_x1_5, MobileViTV2_x1_75, MobileViTV2_x2_0
from .model_zoo.tinynet import TinyNet_A, TinyNet_B, TinyNet_C, TinyNet_D, TinyNet_E
from .model_zoo.mobilevit_v3 import MobileViTV3_XXS, MobileViTV3_XS, MobileViTV3_S, MobileViTV3_XXS_L2, MobileViTV3_XS_L2, MobileViTV3_S_L2, MobileViTV3_x0_5, MobileViTV3_x0_75, MobileViTV3_x1_0
from .model_zoo.svtrnet import SVTR_tiny, SVTR_base, SVTR_large

from .variant_models.resnet_variant import ResNet50_last_stage_stride1
from .variant_models.resnet_variant import ResNet50_adaptive_max_pool2d
from .variant_models.resnet_variant import ResNet50_metabin
from .variant_models.vgg_variant import VGG19Sigmoid
from .variant_models.pp_lcnet_variant import PPLCNet_x2_5_Tanh
from .variant_models.pp_lcnetv2_variant import PPLCNetV2_base_ShiTu
from .variant_models.efficientnet_variant import EfficientNetB3_watermark
from .variant_models.foundation_vit_variant import CLIP_large_patch14_224_aesthetic
from .variant_models.swin_transformer_variant import SwinTransformer_tiny_patch4_window7_224_SOLIDER, SwinTransformer_small_patch4_window7_224_SOLIDER, SwinTransformer_base_patch4_window7_224_SOLIDER
from .model_zoo.adaface_ir_net import AdaFace_IR_18, AdaFace_IR_34, AdaFace_IR_50, AdaFace_IR_101, AdaFace_IR_152, AdaFace_IR_SE_50, AdaFace_IR_SE_101, AdaFace_IR_SE_152, AdaFace_IR_SE_200
from .model_zoo.wideresnet import WideResNet
from .model_zoo.uniformer import UniFormer_small, UniFormer_small_plus, UniFormer_small_plus_dim64, UniFormer_base, UniFormer_base_ls


# help whl get all the models' api (class type) and components' api (func type)
def get_apis():
    current_func = sys._getframe().f_code.co_name
    current_module = sys.modules[__name__]
    api = []
    for _, obj in inspect.getmembers(current_module,
                                     inspect.isclass) + inspect.getmembers(
                                         current_module, inspect.isfunction):
        api.append(obj.__name__)
    api.remove(current_func)
    return api


__all__ = get_apis()
