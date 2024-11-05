# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

# This code is based on AdaFace(https://github.com/mk-minchul/AdaFace)
# Paper: AdaFace: Quality Adaptive Margin for Face Recognition
from paddle.nn import Layer
import math
import paddle


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.divide(input, norm)
    return output


class AdaMargin(Layer):
    def __init__(
            self,
            embedding_size=512,
            class_num=70722,
            m=0.4,
            h=0.333,
            s=64.,
            t_alpha=1.0, ):
        super(AdaMargin, self).__init__()
        self.classnum = class_num
        kernel_weight = paddle.uniform(
            [embedding_size, class_num], min=-1, max=1)
        kernel_weight_norm = paddle.norm(
            kernel_weight, p=2, axis=0, keepdim=True)
        kernel_weight_norm = paddle.where(kernel_weight_norm > 1e-5,
                                          kernel_weight_norm,
                                          paddle.ones_like(kernel_weight_norm))
        kernel_weight = kernel_weight / kernel_weight_norm
        self.kernel = self.create_parameter(
            [embedding_size, class_num],
            attr=paddle.nn.initializer.Assign(kernel_weight))

        # initial kernel
        # self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', paddle.zeros([1]), persistable=True)
        self.register_buffer(
            'batch_mean', paddle.ones([1]) * 20, persistable=True)
        self.register_buffer(
            'batch_std', paddle.ones([1]) * 100, persistable=True)

    def forward(self, embbedings, label):
        if not self.training:
            return embbedings

        norms = paddle.norm(embbedings, 2, 1, True)
        embbedings = paddle.divide(embbedings, norms)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = paddle.mm(embbedings, kernel_norm)
        cosine = paddle.clip(cosine, -1 + self.eps,
                             1 - self.eps)  # for stability

        safe_norms = paddle.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with paddle.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha
                                                     ) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha
                                                   ) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (
            self.batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = paddle.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = paddle.nn.functional.one_hot(
            label.reshape([-1]), self.classnum)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = paddle.acos(cosine)
        theta_m = paddle.clip(
            theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = paddle.cos(theta_m)

        # g_additive
        m_cos = paddle.nn.functional.one_hot(
            label.reshape([-1]), self.classnum)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
