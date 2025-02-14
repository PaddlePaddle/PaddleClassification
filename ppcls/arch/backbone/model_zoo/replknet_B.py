# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RepLKNet in Paddle
A Paddle Impelementation of RepLKNet as described in:
"Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs code"
    - Paper Link: https://arxiv.org/abs/2203.06717
"""
import os
import paddle
import paddle.nn as nn

import sys


# os.environ['LARGE_KERNEL_CONV_IMPL']='true'
class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        #random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = keep_prob + paddle.rand(shape, dtype='float32').astype(inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        output = inputs.divide(keep_prob) * random_tensor # divide to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)

def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias_attr):
    """ Return a regular Conv op or an optimized Conv op for large kernel (not supported yet)
    Now only support regular conv op
    """

    conv = nn.Conv2D(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingNormal()),
                     bias_attr=bias_attr)
    return conv


def fuse_bn_tensor(branch, groups=None):
    """ fuse bn into conv
    Args:
        branch(ConvNormAct): nn.Sequential(conv2d -> norm -> act).
        groups(int): gropus in branch's conv op, default: None
    Returns:
        kernel_weight(tensor), kernel_bias(tensor): fused conv weights value and bias value
    """
    if branch is None:
        return 0, 0
    if isinstance(branch, ConvNormAct):
        kernel = branch.conv.weight
        bias = branch.conv.bias
        assert bias is None
        running_mean = branch.norm._mean
        running_var = branch.norm._variance
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm._epsilon

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, -running_mean * gamma / std + beta
    else:
        raise ValueError("fuse_bn_tensor only supports ConvActNorm")


class ConvNormAct(nn.Sequential):
    """Layer ops: Conv2D -> NormLayer -> ActLayer"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2D):
        layers = [('conv', get_conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      groups=groups,
                                      dilation=1,
                                      bias_attr=bias_attr))]
        if norm_layer is not None:
            layers.append(('bn', norm_layer(out_channels)))
        if act_layer is not None:
            layers.append(('nonlinear', act_layer()))

        super().__init__(*layers)


class ReparamLargeKernelConv(nn.Layer):
    """Large Kernel Conv Block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups,
                 small_kernel,
                 small_kernel_merged=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=1,
                                          groups=groups,
                                          bias_attr=paddle.ParamAttr(
                                              initializer=nn.initializer.Constant(0.0)),
                                          )
        else:
            # large conv + bn
            self.lkb_origin = ConvNormAct(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          groups=groups,
                                          act_layer=None)
            if small_kernel is not None:
                assert small_kernel <= kernel_size
                # small conv + bn
                self.small_conv = ConvNormAct(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=small_kernel,
                                              stride=stride,
                                              padding=small_kernel // 2,
                                              groups=groups,
                                              act_layer=None)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(x)
        return out

    def get_equivalent_kernel_bias(self):
        """Get reparam kernel weights and bias
        First get fused bn kernel weights and bias for large and small kernels
        Then reparam small kernel into large kernel and get reparamed large kernel weight and bias
        """
        large_kernel_weight, large_kernel_bias = fuse_bn_tensor(self.large_kernel_origin)
        if hasattr(self, 'small_kernel_conv'):
            small_kernel_weight, small_kernel_bias = fuse_bn_tensor(self.small_kernel_conv)
            large_kernel_weight += nn.functional.pad(small_kernel_weight,
                                                     [(self.kernel_size - self.small_kernel) // 2] * 4)
            large_kernel_bias += small_kernel_bias
        return large_kernel_weight, large_kernel_bias

    def merge_kernel(self):
        """Get reparamed large kernel weight and bias and set to large conv op"""
        large_kernel_weight, large_kernel_bias = self.get_equivalent_kernel_bias()
        self.large_kernel_reparam = get_conv2d(in_channels=self.large_kernel_origin.conv._in_channels,
                                               out_channels=self.large_kernel_origin.conv._out_channels,
                                               kernel_size=self.large_kernel_origin.conv._kernel_size,
                                               stride=self.large_kernel_origin.conv._stride,
                                               padding=self.large_kernel_origin.conv._padding,
                                               dilation=self.large_kernel_origin.conv._dilation,
                                               groups=self.large_kernel_origin.conv._groups,
                                               bias_attr=paddle.ParamAttr(
                                                   initializer=nn.initializer.Constant(0.0)),
                                               )
        self.large_kernel_reparam.weight.set_value(large_kernel_weight)
        self.large_kernel_reparam.bias.set_value(large_kernel_bias)
        self.__delattr__('large_kernel_origin')
        if hasattr(self, 'small_kernel_conv'):
            self.__delattr__('small_kernel_conv')


class RepLKBlock(nn.Layer):
    """RepLKBlock in each stage"""

    def __init__(self,
                 in_channels,
                 dw_channels,
                 block_large_kernel_size,
                 small_kernel,
                 droppath,
                 small_kernel_merged=False):
        super().__init__()
        # 1x1 pointwise conv + bn + relu
        self.pw1 = ConvNormAct(in_channels=in_channels,
                               out_channels=dw_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1)
        # 1x1 pointwise conv + bn
        self.pw2 = ConvNormAct(in_channels=dw_channels,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               act_layer=None)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels,
                                                   out_channels=dw_channels,
                                                   kernel_size=block_large_kernel_size,
                                                   stride=1,
                                                   groups=dw_channels,
                                                   small_kernel=small_kernel,
                                                   small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = nn.BatchNorm2D(in_channels)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

    def forward(self, x):
        h = x
        x = self.prelkb_bn(x)
        x = self.pw1(x)
        x = self.large_kernel(x)
        x = self.lk_nonlinear(x)
        x = self.pw2(x)
        x = self.drop_path(x)
        x = h + x
        return x


class ConvFFN(nn.Layer):
    """ConvFFN block in each stage"""

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 droppath):
        super().__init__()
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.preffn_bn = nn.BatchNorm2D(in_channels)
        # 1x1 pointwise conv + bn + relu
        self.pw1 = ConvNormAct(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               act_layer=None)
        # 1x1 pointwise conv + bn
        self.pw2 = ConvNormAct(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               act_layer=None)
        self.act = nn.GELU()

    def forward(self, x):
        h = x
        x = self.preffn_bn(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop_path(x)
        x = h + x
        return x


class RepLKNetStage(nn.Layer):
    def __init__(self,
                 channels,
                 num_blocks,
                 stage_large_kernel_size,
                 small_kernel,
                 droppath,
                 dw_ratio=1,
                 ffn_ratio=4,
                 small_kernel_merged=False,
                 norm_inter_features=False):
        super().__init__()
        self.blocks = nn.LayerList()
        for i in range(num_blocks):
            block_droppath = droppath[i] if isinstance(droppath, list) else droppath
            replk_block = RepLKBlock(in_channels=channels,
                                     dw_channels=int(channels * dw_ratio),
                                     block_large_kernel_size=stage_large_kernel_size,
                                     small_kernel=small_kernel,
                                     droppath=block_droppath,
                                     small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels,
                                    hidden_channels=int(channels * ffn_ratio),
                                    out_channels=channels,
                                    droppath=block_droppath)
            self.blocks.append(replk_block)
            self.blocks.append(convffn_block)
        self.norm = nn.BatchNorm2D(channels) if norm_inter_features else nn.Identity()

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return x


class RepLKNet(nn.Layer):
    def __init__(self,
                 large_kernel_sizes,
                 layers,  # [2,2,18,2
                 channels,
                 droppath,
                 small_kernel,
                 dw_ratio=1,
                 ffn_ratio=4,
                 in_channels=3,
                 num_classes=1000,
                 out_indices=None,
                 small_kernel_merged=False,
                 norm_inter_features=False):
        super().__init__()
        assert num_classes is None or out_indices is None
        assert num_classes is not None or out_indices is not None

        self.num_stages = len(layers)
        self.out_indices = out_indices
        self.norm_inter_features = norm_inter_features
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, sum(layers))]

        # Stem layers
        self.stem = nn.Sequential(
            # 3x3 conv w/t stride 2
            ConvNormAct(in_channels=in_channels,
                        out_channels=channels[0],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=1),
            # 3x3 depthwise conv
            ConvNormAct(in_channels=channels[0],
                        out_channels=channels[0],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=channels[0]),
            # 1x1 pointwise conv
            ConvNormAct(in_channels=channels[0],
                        out_channels=channels[0],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1),
            # 3x3 depthwise conv w/t stride 2
            ConvNormAct(in_channels=channels[0],
                        out_channels=channels[0],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=channels[0]))
        # Stage + Transition layers
        self.stages = nn.LayerList()
        self.transitions = nn.LayerList()
        for stage_idx in range(self.num_stages):
            self.stages.append(RepLKNetStage(channels=channels[stage_idx],
                                             num_blocks=layers[stage_idx],
                                             stage_large_kernel_size=large_kernel_sizes[stage_idx],
                                             small_kernel=small_kernel,
                                             small_kernel_merged=small_kernel_merged,
                                             droppath=depth_decay[sum(layers[:stage_idx]): sum(layers[:stage_idx + 1])],
                                             dw_ratio=dw_ratio,
                                             ffn_ratio=ffn_ratio,
                                             norm_inter_features=norm_inter_features))
            if stage_idx < self.num_stages - 1:
                self.transitions.append(
                    nn.Sequential(
                        # 1x1 pointwise conv
                        ConvNormAct(in_channels=channels[stage_idx],
                                    out_channels=channels[stage_idx + 1],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1),
                        # 3x3 depthwise conv w/t stride 2
                        ConvNormAct(in_channels=channels[stage_idx + 1],
                                    out_channels=channels[stage_idx + 1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=channels[stage_idx + 1])))
        # Head layers
        if num_classes is not None:
            self.norm = nn.BatchNorm2D(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2D(1)
            self.head = nn.Linear(in_features=channels[-1],
                                  out_features=num_classes,
                                  weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingNormal()),
                                  bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)))

    def forward_features(self, x):
        x = self.stem(x)
        if self.out_indices is None:  # only output the final layer
            x = self.stages[0](x)
            for stage, transition in zip(self.stages[1:], self.transitions):
                x = transition(x)
                x = stage(x)

            return x
        else:
            outs = []
            x = self.stages[0](x)
            if 0 in self.out_indices:
                outs.append(self.stages[0].norm(x))
            for idx, (stage, transition) in enumerate(zip(self.stages[1:], self.transitions)):
                x = transition(x)
                x = stage(x)
                if idx in self.out_indices:
                    outs.append(stage.norm(x))
            return outs

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.head(x)
        return x

    def structural_reparam(self):
        for layer in self.sublayers():
            if hasattr(layer, 'merge_kernel'):
                layer.merge_kernel()


def replknet(class_num):
    """Build RepLKNet by reading options in config object
    Args:
        config: config instance contains setting options
    Returns:
        model: RepLKNet model
    """
    model = RepLKNet(large_kernel_sizes=[31,29,27,13],
                     layers=[2,2,18,2],
                     channels=[128,256,512,1024],
                     droppath=0.5,
                     small_kernel=5,
                     dw_ratio=1.0,
                     ffn_ratio=4.0,
                     in_channels=3,
                     num_classes=class_num,
                     out_indices=None,
                     small_kernel_merged=False,
                     norm_inter_features=False)
    # if sync_bn:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model
