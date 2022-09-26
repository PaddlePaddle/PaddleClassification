# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import paddle


class CrossBatchMemory(object):
    def __init__(self, size: int, feat_dim: int):
        self.size = size
        self.feat_dim = feat_dim
        self.feats = paddle.zeros([self.size, self.feat_dim])
        self.targets = paddle.zeros([self.size, ], dtype="int64")
        self.ptr = 0
        self.cur_size = 0

    @property
    def is_full(self) -> bool:
        return self.cur_size >= self.size

    def get(self) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """return features and targets in memory bank

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: [features, targets]
        """
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats: paddle.Tensor, targets: paddle.Tensor) -> None:
        """put newest feats and targets into memory bank and pop oldest feats and targets from momory bank

        Args:
            feats (paddle.Tensor): features to enque
            targets (paddle.Tensor): targets to enque
        """
        input_size = len(targets)
        if self.ptr + input_size > self.size:
            self.feats[-input_size:] = feats
            self.targets[-input_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + input_size] = feats
            self.targets[self.ptr: self.ptr + input_size] = targets
            self.ptr += input_size
        self.cur_size += input_size
