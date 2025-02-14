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

# reference: https://arxiv.org/abs/1409.4842

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "GoogLeNet":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GoogLeNet_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


def xavier(channels, filter_size, name):
    stdv = (3.0 / (filter_size**2 * channels))**0.5
    param_attr = ParamAttr(
        initializer=Uniform(-stdv, stdv), name=name + "_weights")
    return param_attr


class ConvLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None,
                 data_format="NCHW",
                 insert_bn=True):
        super(ConvLayer, self).__init__()

        self.insert_bn = insert_bn
        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            data_format=data_format)
        if insert_bn:
            self._bn = BatchNorm(
                num_channels=num_filters,
                param_attr=ParamAttr(name=name + "_bn"))

    def forward(self, inputs):
        y = self._conv(inputs)
        if self.insert_bn:
            y = self._bn(y)
        return y


class Inception(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter1,
                 filter3R,
                 filter3,
                 filter5R,
                 filter5,
                 proj,
                 name=None,
                 data_format="NCHW",
                 insert_bn=True):
        super(Inception, self).__init__()
        self.data_format = data_format

        self._conv1 = ConvLayer(
            input_channels,
            filter1,
            1,
            name="inception_" + name + "_1x1",
            data_format=data_format,
            insert_bn=insert_bn)
        self._conv3r = ConvLayer(
            input_channels,
            filter3R,
            1,
            name="inception_" + name + "_3x3_reduce",
            data_format=data_format,
            insert_bn=insert_bn)
        self._conv3 = ConvLayer(
            filter3R,
            filter3,
            3,
            name="inception_" + name + "_3x3",
            data_format=data_format,
            insert_bn=insert_bn)
        self._conv5r = ConvLayer(
            input_channels,
            filter5R,
            1,
            name="inception_" + name + "_5x5_reduce",
            data_format=data_format,
            insert_bn=insert_bn)
        self._conv5 = ConvLayer(
            filter5R,
            filter5,
            5,
            name="inception_" + name + "_5x5",
            data_format=data_format,
            insert_bn=insert_bn)
        self._pool = MaxPool2D(
            kernel_size=3, stride=1, padding=1, data_format=data_format)

        self._convprj = ConvLayer(
            input_channels,
            proj,
            1,
            name="inception_" + name + "_3x3_proj",
            data_format=data_format,
            insert_bn=insert_bn)

    def forward(self, inputs):
        conv1 = self._conv1(inputs)

        conv3r = self._conv3r(inputs)
        conv3 = self._conv3(conv3r)

        conv5r = self._conv5r(inputs)
        conv5 = self._conv5(conv5r)

        pool = self._pool(inputs)
        convprj = self._convprj(pool)

        if self.data_format == "NHWC":
            cat = paddle.concat([conv1, conv3, conv5, convprj], axis=3)
        else:
            cat = paddle.concat([conv1, conv3, conv5, convprj], axis=1)
        cat = F.relu(cat)
        return cat


class GoogLeNetDY(nn.Layer):
    def __init__(self, class_num=1000, data_format="NCHW", insert_bn=True):
        super(GoogLeNetDY, self).__init__()
        self.data_format = data_format
        self._conv = ConvLayer(
            3,
            64,
            7,
            2,
            name="conv1",
            data_format=data_format,
            insert_bn=insert_bn)
        self._pool = MaxPool2D(
            kernel_size=3, stride=2, data_format=data_format)
        self._conv_1 = ConvLayer(
            64,
            64,
            1,
            name="conv2_1x1",
            data_format=data_format,
            insert_bn=insert_bn)
        self._conv_2 = ConvLayer(
            64,
            192,
            3,
            name="conv2_3x3",
            data_format=data_format,
            insert_bn=insert_bn)

        self._ince3a = Inception(
            192,
            192,
            64,
            96,
            128,
            16,
            32,
            32,
            name="ince3a",
            data_format=data_format,
            insert_bn=insert_bn)
        self._ince3b = Inception(
            256,
            256,
            128,
            128,
            192,
            32,
            96,
            64,
            name="ince3b",
            data_format=data_format,
            insert_bn=insert_bn)

        self._ince4a = Inception(
            480,
            480,
            192,
            96,
            208,
            16,
            48,
            64,
            name="ince4a",
            data_format=data_format,
            insert_bn=insert_bn)
        self._ince4b = Inception(
            512,
            512,
            160,
            112,
            224,
            24,
            64,
            64,
            name="ince4b",
            data_format=data_format,
            insert_bn=insert_bn)
        self._ince4c = Inception(
            512,
            512,
            128,
            128,
            256,
            24,
            64,
            64,
            name="ince4c",
            data_format=data_format,
            insert_bn=insert_bn)
        self._ince4d = Inception(
            512,
            512,
            112,
            144,
            288,
            32,
            64,
            64,
            name="ince4d",
            data_format=data_format,
            insert_bn=insert_bn)
        self._ince4e = Inception(
            528,
            528,
            256,
            160,
            320,
            32,
            128,
            128,
            name="ince4e",
            data_format=data_format,
            insert_bn=insert_bn)

        self._ince5a = Inception(
            832,
            832,
            256,
            160,
            320,
            32,
            128,
            128,
            name="ince5a",
            data_format=data_format,
            insert_bn=insert_bn)
        self._ince5b = Inception(
            832,
            832,
            384,
            192,
            384,
            48,
            128,
            128,
            name="ince5b",
            data_format=data_format,
            insert_bn=insert_bn)

        self._pool_5 = AdaptiveAvgPool2D(1, data_format=data_format)

        self._drop = Dropout(p=0.4, mode="downscale_in_infer")
        self.flatten = nn.Flatten()
        self._fc_out = Linear(
            1024,
            class_num,
            weight_attr=xavier(1024, 1, "out"),
            bias_attr=ParamAttr(name="out_offset"))
        self._pool_o1 = AvgPool2D(
            kernel_size=5, stride=3, data_format=data_format)
        self._conv_o1 = ConvLayer(
            512,
            128,
            1,
            name="conv_o1",
            data_format=data_format,
            insert_bn=insert_bn)
        self._fc_o1 = Linear(
            1152,
            1024,
            weight_attr=xavier(2048, 1, "fc_o1"),
            bias_attr=ParamAttr(name="fc_o1_offset"))
        self._drop_o1 = Dropout(p=0.7, mode="downscale_in_infer")
        self._out1 = Linear(
            1024,
            class_num,
            weight_attr=xavier(1024, 1, "out1"),
            bias_attr=ParamAttr(name="out1_offset"))
        self._pool_o2 = AvgPool2D(
            kernel_size=5, stride=3, data_format=data_format)
        self._conv_o2 = ConvLayer(
            528,
            128,
            1,
            name="conv_o2",
            data_format=data_format,
            insert_bn=insert_bn)
        self._fc_o2 = Linear(
            1152,
            1024,
            weight_attr=xavier(2048, 1, "fc_o2"),
            bias_attr=ParamAttr(name="fc_o2_offset"))
        self._drop_o2 = Dropout(p=0.7, mode="downscale_in_infer")
        self._out2 = Linear(
            1024,
            class_num,
            weight_attr=xavier(1024, 1, "out2"),
            bias_attr=ParamAttr(name="out2_offset"))

    def forward(self, inputs):
        if self.data_format == "NHWC":
            inputs = paddle.transpose(inputs, [0, 2, 3, 1])
            inputs.stop_gradient = True
        x = self._conv(inputs)
        x = self._pool(x)
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._pool(x)

        x = self._ince3a(x)
        x = self._ince3b(x)
        x = self._pool(x)

        ince4a = self._ince4a(x)
        x = self._ince4b(ince4a)
        x = self._ince4c(x)
        ince4d = self._ince4d(x)
        x = self._ince4e(ince4d)
        x = self._pool(x)

        x = self._ince5a(x)
        ince5b = self._ince5b(x)

        x = self._pool_5(ince5b)
        x = self._drop(x)
        x = self.flatten(x)
        out = self._fc_out(x)

        x = self._pool_o1(ince4a)
        x = self._conv_o1(x)
        x = self.flatten(x)
        x = self._fc_o1(x)
        x = F.relu(x)
        x = self._drop_o1(x)
        out1 = self._out1(x)

        x = self._pool_o2(ince4d)
        x = self._conv_o2(x)
        x = self.flatten(x)
        x = self._fc_o2(x)
        x = self._drop_o2(x)
        out2 = self._out2(x)
        return [out, out1, out2]


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def GoogLeNet(pretrained=False, use_ssld=False, **kwargs):
    model = GoogLeNetDY(**kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["GoogLeNet"], use_ssld=use_ssld)
    return model
