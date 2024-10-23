# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal, Constant
from paddle.nn import Conv2D, BatchNorm2D, PReLU, AdaptiveAvgPool2D, MaxPool2D
from paddle.regularizer import L2Decay
from paddle import ParamAttr

from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "PPHGNetV2_B0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B0_ssld_pretrained.pdparams",
    "PPHGNetV2_B1":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B1_ssld_pretrained.pdparams",
    "PPHGNetV2_B2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B2_ssld_pretrained.pdparams",
    "PPHGNetV2_B3":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B3_ssld_pretrained.pdparams",
    "PPHGNetV2_B4":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B4_ssld_pretrained.pdparams",
    "PPHGNetV2_B5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B5_ssld_pretrained.pdparams",
    "PPHGNetV2_B6":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B6_ssld_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

kaiming_normal_ = KaimingNormal()
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class LearnableAffineBlock(TheseusLayer):
    """  
    Create a learnable affine block module. This module can significantly improve accuracy on smaller models.

    Args:  
        scale_value (float): The initial value of the scale parameter, default is 1.0.  
        bias_value (float): The initial value of the bias parameter, default is 0.0.  
        lr_mult (float): The learning rate multiplier, default is 1.0.  
        lab_lr (float): The learning rate, default is 0.01.  
    """

    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0,
                 lr_mult=1.0,
                 lab_lr=0.01):
        super().__init__()
        self.scale = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("scale", self.scale)
        self.bias = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(TheseusLayer):
    """  
    ConvBNAct is a combination of convolution and batchnorm layers.
  
    Args:  
        in_channels (int): Number of input channels.  
        out_channels (int): Number of output channels.  
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int/str): Padding or padding type for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        use_act: (bool): Whether to use activation function. Defaults to True.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.  
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.  
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 use_act=True,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding
            if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=False)
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult))
        if self.use_act:
            self.act = PReLU(out_channels)
            if self.use_lab:
                self.lab = LearnableAffineBlock(lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(TheseusLayer):
    """  
    LightConvBNAct is a combination of pw and dw layers.
  
    Args:  
        in_channels (int): Number of input channels.  
        out_channels (int): Number of output channels.  
        kernel_size (int): Size of the depth-wise convolution kernel.  
        use_lab (bool): Whether to use the LAB operation. Defaults to False.  
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.  
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_lab=False,
                 lr_mult=1.0,
                 **kwargs):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab,
            lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(TheseusLayer):
    """  
    StemBlock for PP-HGNetV2.
  
    Args:  
        in_channels (int): Number of input channels.  
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.  
        use_lab (bool): Whether to use the LAB operation. Defaults to False.  
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.  
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.pool = nn.MaxPool2D(
            kernel_size=2, stride=1, ceil_mode=True, padding="SAME")

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = paddle.concat([x1, x2], 1)
        x = self.stem3(x)
        x = self.stem4(x)

        return x


class HGV2_Block(TheseusLayer):
    """  
    HGV2_Block, the basic unit that constitutes the HGV2_Stage.
  
    Args:  
        in_channels (int): Number of input channels. 
        mid_channels (int): Number of middle channels.  
        out_channels (int): Number of output channels.  
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        layer_num (int): Number of layers in the HGV2 block. Defaults to 6.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int/str): Padding or padding type for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        use_act (bool): Whether to use activation function. Defaults to True.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.  
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.  
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 layer_num=6,
                 identity=False,
                 light_block=True,
                 use_lab=False,
                 lr_mult=1.0):
        super().__init__()
        self.identity = identity

        self.layers = nn.LayerList()
        block_type = "LightConvBNAct" if light_block else "ConvBNAct"
        for i in range(layer_num):
            self.layers.append(
                eval(block_type)(in_channels=in_channels
                                 if i == 0 else mid_channels,
                                 out_channels=mid_channels,
                                 stride=1,
                                 kernel_size=kernel_size,
                                 use_lab=use_lab,
                                 lr_mult=lr_mult))
        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = paddle.concat(output, axis=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x += identity
        return x


class HGV2_Stage(TheseusLayer):
    """  
    HGV2_Stage, the basic unit that constitutes the PPHGNetV2.
  
    Args:  
        in_channels (int): Number of input channels. 
        mid_channels (int): Number of middle channels.  
        out_channels (int): Number of output channels.  
        block_num (int): Number of blocks in the HGV2 stage. 
        layer_num (int): Number of layers in the HGV2 block. Defaults to 6.
        is_downsample (bool): Whether to use downsampling operation. Defaults to False.
        light_block (bool): Whether to use light block. Defaults to True.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        use_lab (bool, optional): Whether to use the LAB operation. Defaults to False.  
        lr_mult (float, optional): Learning rate multiplier for the layer. Defaults to 1.0.  
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num=6,
                 is_downsample=True,
                 light_block=True,
                 kernel_size=3,
                 use_lab=False,
                 lr_mult=1.0):

        super().__init__()
        self.is_downsample = is_downsample
        if self.is_downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,
                lr_mult=lr_mult)

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HGV2_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab,
                    lr_mult=lr_mult))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.is_downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNetV2(TheseusLayer):
    """
    PPHGNetV2

    Args:
        stage_config (dict): Config for PPHGNetV2 stages. such as the number of channels, stride, etc.
        stem_channels: (list): Number of channels of the stem of the PPHGNetV2.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.  
        use_last_conv (bool): Whether to use the last conv layer as the output channel. Defaults to True.
        class_expand (int): Number of channels for the last 1x1 convolutional layer.
        drop_prob (float): Dropout probability for the last 1x1 convolutional layer. Defaults to 0.0.
        class_num (int): The number of classes for the classification layer. Defaults to 1000.
        lr_mult_list (list): Learning rate multiplier for the stages. Defaults to [1.0, 1.0, 1.0, 1.0, 1.0].
    Returns:
        model: nn.Layer. Specific PPHGNetV2 model depends on args.
    """

    def __init__(self,
                 stage_config,
                 stem_channels=[3, 32, 64],
                 use_lab=False,
                 use_last_conv=True,
                 class_expand=2048,
                 dropout_prob=0.0,
                 class_num=1000,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
                 **kwargs):
        super().__init__()
        self.use_lab = use_lab
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.class_num = class_num

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab,
            lr_mult=lr_mult_list[0])

        # stages
        self.stages = nn.LayerList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, is_downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HGV2_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    is_downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    lr_mult=lr_mult_list[i + 1]))

        self.avg_pool = AdaptiveAvgPool2D(1)

        if self.use_last_conv:
            self.last_conv = Conv2D(
                in_channels=out_channels,
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.act = PReLU(self.class_expand)
            if self.use_lab:
                self.lab = LearnableAffineBlock()
            self.dropout = nn.Dropout(
                p=dropout_prob, mode="downscale_in_infer")

        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = nn.Linear(self.class_expand if self.use_last_conv else
                            out_channels, self.class_num)

        self._init_weights()

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2D)):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)
        x = self.avg_pool(x)

        if self.use_last_conv:
            x = self.last_conv(x)
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(
            model,
            model_url,
            use_ssld=use_ssld,
            use_ssld_stage1_pretrained=True)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def PPHGNetV2Face_B0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B0
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B0` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
    }

    model = PPHGNetV2(
        stem_channels=[3, 16, 16],
        stage_config=stage_config,
        use_lab=True,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPHGNetV2_B0"], use_ssld)
    return model


def PPHGNetV2Face_B1(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B1
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B1` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, True, False, 3, 3],
        "stage2": [64, 48, 256, 1, False, False, 3, 3],
        "stage3": [256, 96, 512, 2, False, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32],
        stage_config=stage_config,
        use_lab=True,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPHGNetV2_B1"], use_ssld)
    return model


def PPHGNetV2Face_B2(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B2
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B2` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 96, 1, False, False, 3, 4],
        "stage2": [96, 64, 384, 1, True, False, 3, 4],
        "stage3": [384, 128, 768, 3, True, True, 5, 4],
        "stage4": [768, 256, 1536, 1, True, True, 5, 4],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32],
        stage_config=stage_config,
        use_lab=True,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPHGNetV2_B2"], use_ssld)
    return model


def PPHGNetV2Face_B3(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B3
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B3` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 128, 1, False, False, 3, 5],
        "stage2": [128, 64, 512, 1, True, False, 3, 5],
        "stage3": [512, 128, 1024, 3, True, True, 5, 5],
        "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32],
        stage_config=stage_config,
        use_lab=True,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPHGNetV2_B3"], use_ssld)
    return model


def PPHGNetV2Face_B4(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B4
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B4` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [48, 48, 128, 1, False, False, 3, 6],
        "stage2": [128, 96, 512, 1, True, False, 3, 6],
        "stage3": [512, 192, 1024, 3, True, True, 5, 6],
        "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
    }

    model = PPHGNetV2(
        stem_channels=[3, 32, 48],
        stage_config=stage_config,
        use_lab=False,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPHGNetV2_B4"], use_ssld)
    return model


def PPHGNetV2Face_B5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B5
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B5` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [64, 64, 128, 1, False, False, 3, 6],
        "stage2": [128, 128, 512, 2, True, False, 3, 6],
        "stage3": [512, 256, 1024, 5, True, True, 5, 6],
        "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
    }

    model = PPHGNetV2(
        stem_channels=[3, 32, 64],
        stage_config=stage_config,
        use_lab=False,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPHGNetV2_B5"], use_ssld)
    return model


def PPHGNetV2Face_B6(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B6
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B6` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [96, 96, 192, 2, False, False, 3, 6],
        "stage2": [192, 192, 512, 3, True, False, 3, 6],
        "stage3": [512, 384, 1024, 6, True, True, 5, 6],
        "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
    }

    model = PPHGNetV2(
        stem_channels=[3, 48, 96],
        stage_config=stage_config,
        use_lab=False,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPHGNetV2_B6"], use_ssld)
    return model