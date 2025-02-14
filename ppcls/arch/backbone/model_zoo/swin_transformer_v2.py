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

# Code was based on https://github.com/microsoft/Swin-Transformer
# reference: https://arxiv.org/abs/2111.09883

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
import numpy as np
import math

from .vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity
from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "SwinTransformerV2_tiny_patch4_window8_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_tiny_patch4_window8_256_pretrained.pdparams",
    "SwinTransformerV2_tiny_patch4_window16_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_tiny_patch4_window16_256_pretrained.pdparams",
    "SwinTransformerV2_small_patch4_window8_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_small_patch4_window8_256_pretrained.pdparams",
    "SwinTransformerV2_small_patch4_window16_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_small_patch4_window16_256_pretrained.pdparams",
    "SwinTransformerV2_base_patch4_window8_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_base_patch4_window8_256_pretrained.pdparams",
    "SwinTransformerV2_base_patch4_window16_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_base_patch4_window16_256_pretrained.pdparams",
    "SwinTransformerV2_base_patch4_window24_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_base_patch4_window24_384_pretrained.pdparams",
    "SwinTransformerV2_large_patch4_window16_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_large_patch4_window16_256_pretrained.pdparams",
    "SwinTransformerV2_large_patch4_window24_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformerV2_large_patch4_window24_384_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def pading_for_not_divisible(pixel_values,
                             height,
                             width,
                             patch_size,
                             format="BCHW",
                             function="split"):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if function == "split":
        pading_width = patch_size[1] - width % patch_size[1]
        pading_height = patch_size[0] - height % patch_size[0]
    elif function == "merge":
        pading_width = width % 2
        pading_height = height % 2
    if format == "BCHW":
        pad_index = (0, 0, 0, 0, 0, pading_height, 0, pading_width)
    elif format == "BHWC":
        pad_index = (0, 0, 0, pading_height, 0, pading_width, 0, 0)
    else:
        assert ("vaild format")

    return F.pad(pixel_values, pad_index), pad_index


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose(perm=[0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, C])
    return windows


def pad_patch(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose(perm=[0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = x.transpose(perm=[0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])
    return x


class WindowAttention(nn.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = self.create_parameter(
            [num_heads, 1, 1],
            dtype='float32',
            default_initializer=Constant(math.log(10.)))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(
                2, 512, bias_attr=True),
            nn.ReLU(),
            nn.Linear(
                512, num_heads, bias_attr=False))

        # get relative_coords_table
        relative_coords_h = paddle.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype='float32')
        relative_coords_w = paddle.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype='float32')
        relative_coords_table = paddle.stack(
            paddle.meshgrid([relative_coords_h, relative_coords_w])).transpose(
                perm=[1, 2, 0]).unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (
                pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (
                pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = paddle.sign(
            relative_coords_table) * paddle.log2(
                paddle.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid(
            [coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(
            perm=[1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[
            0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=False)
        if qkv_bias:
            self.q_bias = self.create_parameter(
                [dim], dtype='float32', default_initializer=zeros_)
            self.v_bias = self.create_parameter(
                [dim], dtype='float32', default_initializer=zeros_)
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = paddle.concat(
                x=[self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(shape=[
            B_, N, 3, self.num_heads, qkv.shape[-1] // (3 * self.num_heads)
        ]).transpose(perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make paddlescript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(
            q, axis=-1) @F.normalize(
                k, axis=-1).transpose(perm=[0, 1, 3, 2]))
        logit_scale = paddle.clip(
            self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).reshape([-1, self.num_heads])
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.reshape([-1])].reshape([
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1
            ])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * F.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N
                                 ]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @v).transpose(perm=[0, 2, 1, 3]).reshape(shape=[B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Layer):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=8,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        """
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = paddle.zeros([1, H, W, 1])  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(
                shape=[-1, self.window_size * self.window_size])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))
        else:
        """
        H, W = self.input_resolution
        attn_mask = paddle.zeros([1, H, W, 1])

        self.register_buffer("attn_mask", attn_mask)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for shifted window multihead self attention
            img_mask = paddle.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None), )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None), )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(
                (-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self, x, input_dimensions):
        H, W = input_dimensions
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = x.reshape([B, H, W, C])

        x, pad_values = pading_for_not_divisible(x, H, W, self.window_size,
                                                 "BHWC")
        _, height_pad, width_pad, _ = x.shape

        padding_state = pad_values[3] > 0 or pad_values[
            5] > 0  # change variable name
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size,
             C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_mask = self.get_attn_mask(height_pad, width_pad, x.dtype)
        attn_windows = self.attn(
            x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(
            [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, height_pad,
                                   width_pad)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                axis=(1, 2))
        else:
            x = shifted_x

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            x = x[:, :H, :W, :]

        x = x.reshape([B, H * W, C])
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Layer):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, input_dimensions):
        """
        x: B, H*W, C
        """
        H, W = input_dimensions
        B, L, C = x.shape
        x = x.reshape([B, H, W, C])
        x, _ = pading_for_not_divisible(x, H, W, 2, "BHWC", function="merge")

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.reshape(x, x.shape)
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = x.reshape([B,  -1, 4 * C])  # B H/2*W/2 4*C
        x = self.reduction(x)
        x = self.norm(x)
        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, input_dimensions):
        H, W = input_dimensions
        for blk in self.blocks:
            x = blk(x, input_dimensions)
        if self.downsample is not None:
            H, W = (H + 1) // 2, (W + 1) // 2
            x = self.downsample(x, input_dimensions)

        return x, (H, W)

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Layer):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 256.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=256,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, 0, 0, 0, 0, 0, 0,
                          self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (
                0,
                0,
                0,
                0,
                0,
                self.patch_size[0] - height % self.patch_size[0],
                0,
                0, )
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        _, _, height, width = x.shape
        output_dimensions = (height, width)
        x = x.flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        # x = self.proj(x).flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, output_dimensions

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (
            self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerV2(nn.Layer):
    r""" Swin TransformerV2
        A PaddlePaddle impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/abs/2111.09883

    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        class_num (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self,
                 img_size=256,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 pretrained_window_sizes=[0, 0, 0, 0],
                 **kwargs):
        super().__init__()

        self.class_num = class_num
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.img_size = img_size
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = self.create_parameter(
                shape=(1, num_patches, embed_dim), default_initializer=zeros_)
            trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.head = nn.Linear(self.num_features,
                              class_num) if class_num > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x, output_dimensions = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x, output_dimensions = layer(x, input_dimensions=output_dimensions)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose([0, 2, 1]))  # B C 1
        x = paddle.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[
            0] * self.patches_resolution[1] // (2**self.num_layers)
        flops += self.num_features * self.class_num
        return flops


def _load_pretrained(pretrained,
                     model,
                     model_url,
                     use_ssld=False,
                     use_imagenet22k_pretrained=False,
                     use_imagenet22kto1k_pretrained=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(
            model,
            model_url,
            use_ssld=use_ssld,
            use_imagenet22k_pretrained=use_imagenet22k_pretrained,
            use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained, **kwargs)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def SwinTransformerV2_tiny_patch4_window8_256(pretrained=False,
                                              use_ssld=False,
                                              **kwargs):
    model = SwinTransformerV2(
        img_size=256,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        drop_path_rate=0.2,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_tiny_patch4_window8_256"],
        use_ssld=use_ssld)
    return model


def SwinTransformerV2_tiny_patch4_window16_256(pretrained=False,
                                               use_ssld=False,
                                               **kwargs):
    model = SwinTransformerV2(
        img_size=256,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        drop_path_rate=0.2,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_tiny_patch4_window16_256"],
        use_ssld=use_ssld)
    return model


def SwinTransformerV2_small_patch4_window8_256(pretrained=False,
                                               use_ssld=False,
                                               **kwargs):
    model = SwinTransformerV2(
        img_size=256,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        drop_path_rate=0.3,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_small_patch4_window8_256"],
        use_ssld=use_ssld)
    return model


def SwinTransformerV2_small_patch4_window16_256(pretrained=False,
                                                use_ssld=False,
                                                **kwargs):
    model = SwinTransformerV2(
        img_size=256,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        drop_path_rate=0.3,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_small_patch4_window16_256"],
        use_ssld=use_ssld)
    return model


def SwinTransformerV2_base_patch4_window8_256(pretrained=False,
                                              use_ssld=False,
                                              **kwargs):
    model = SwinTransformerV2(
        img_size=256,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8,
        drop_path_rate=0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_base_patch4_window8_256"],
        use_ssld=use_ssld)
    return model


def SwinTransformerV2_base_patch4_window16_256(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=False,
        **kwargs):
    model = SwinTransformerV2(
        img_size=256,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=16,
        drop_path_rate=0.5,  # if use imagenet22k or imagenet22kto1k, drop_path_rate=0.2
        **kwargs
    )  # if use imagenet22k, set pretrained_window_sizes=[12, 12, 12, 6]
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_base_patch4_window16_256"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformerV2_base_patch4_window24_384(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=True,
        **kwargs):
    model = SwinTransformerV2(
        img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=24,
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6],
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_base_patch4_window24_384"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformerV2_large_patch4_window16_256(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=True,
        **kwargs):
    model = SwinTransformerV2(
        img_size=256,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=16,
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6],
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_large_patch4_window16_256"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformerV2_large_patch4_window24_384(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=True,
        **kwargs):
    model = SwinTransformerV2(
        img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=24,
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6],
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformerV2_large_patch4_window24_384"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model
