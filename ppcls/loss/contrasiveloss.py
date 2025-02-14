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

from typing import Dict

import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from ppcls.loss.xbm import CrossBatchMemory


class ContrastiveLoss(nn.Layer):
    """ContrastiveLoss

    Args:
        margin (float): margin
        embedding_size (int): number of embedding's dimension
        normalize_feature (bool, optional): whether to normalize embedding. Defaults to True.
        epsilon (float, optional): epsilon. Defaults to 1e-5.
        feature_from (str, optional): which key embedding from input dict. Defaults to "features".
    """

    def __init__(self,
                 margin: float,
                 embedding_size: int,
                 normalize_feature=True,
                 epsilon: float=1e-5,
                 is_text_image_pairs=False,
                 feature_from: str="features"):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.embedding_size = embedding_size
        self.normalize_feature = normalize_feature
        self.epsilon = epsilon
        self.feature_from = feature_from
        self.is_text_image_pairs = is_text_image_pairs
        self.prev_num_logits = 0
        self.labels = {}
        self.world_size = 1
        self.cache_labels = False
        self.rank = 0
        self.local_loss = False

    def text_image_pairs_constrative_loss(self, logits_per_image,
                                          logits_per_text, labels):
        device = logits_per_image.place
        label = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (F.cross_entropy(logits_per_image, label) +
                      F.cross_entropy(logits_per_text, label)) / 2

        return {"Contrastive_loss": total_loss}
    
    def get_ground_truth(self, device, num_logits) -> paddle.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = paddle.arange(num_logits, dtype=paddle.int64)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, input: Dict[str, paddle.Tensor],
                target: paddle.Tensor) -> Dict[str, paddle.Tensor]:

        if self.is_text_image_pairs:
            return self.text_image_pairs_constrative_loss(
                input["image"], input["text"], target)

        feats = input[self.feature_from]
        labels = target

        # normalize along feature dim
        if self.normalize_feature:
            feats = nn.functional.normalize(feats, p=2, axis=1)

        # squeeze labels to shape (batch_size, )
        if labels.ndim >= 2 and labels.shape[-1] == 1:
            labels = paddle.squeeze(labels, axis=[-1])

        loss = self._compute_loss(feats, target, feats, target)

        return {'ContrastiveLoss': loss}

    def _compute_loss(self,
                      inputs_q: paddle.Tensor,
                      targets_q: paddle.Tensor,
                      inputs_k: paddle.Tensor,
                      targets_k: paddle.Tensor) -> paddle.Tensor:
        batch_size = inputs_q.shape[0]
        # Compute similarity matrix
        sim_mat = paddle.matmul(inputs_q, inputs_k.t())

        loss = []
        for i in range(batch_size):
            pos_pair_ = paddle.masked_select(sim_mat[i],
                                             targets_q[i] == targets_k)
            pos_pair_ = paddle.masked_select(pos_pair_,
                                             pos_pair_ < 1 - self.epsilon)

            neg_pair_ = paddle.masked_select(sim_mat[i],
                                             targets_q[i] != targets_k)
            neg_pair = paddle.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = paddle.sum(-pos_pair_ + 1)

            if len(neg_pair) > 0:
                neg_loss = paddle.sum(neg_pair)
            else:
                neg_loss = 0
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / batch_size
        return loss


class ContrastiveLoss_XBM(ContrastiveLoss):
    """ContrastiveLoss with CrossBatchMemory

    Args:
        xbm_size (int): size of memory bank
        xbm_weight (int): weight of CrossBatchMemory's loss
        start_iter (int): store embeddings after start_iter
        margin (float): margin
        embedding_size (int): number of embedding's dimension
        epsilon (float, optional): epsilon. Defaults to 1e-5.
        normalize_feature (bool, optional): whether to normalize embedding. Defaults to True.
        feature_from (str, optional): which key embedding from input dict. Defaults to "features".
    """

    def __init__(self,
                 xbm_size: int,
                 xbm_weight: int,
                 start_iter: int,
                 margin: float,
                 embedding_size: int,
                 epsilon: float=1e-5,
                 normalize_feature=True,
                 feature_from: str="features"):
        super(ContrastiveLoss_XBM, self).__init__(
            margin, embedding_size, normalize_feature, epsilon, feature_from)
        self.xbm = CrossBatchMemory(xbm_size, embedding_size)
        self.xbm_weight = xbm_weight
        self.start_iter = start_iter
        self.iter = 0

    def __call__(self, input: Dict[str, paddle.Tensor],
                 target: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        feats = input[self.feature_from]
        labels = target

        # normalize along feature dim
        if self.normalize_feature:
            feats = nn.functional.normalize(feats, p=2, axis=1)

        # squeeze labels to shape (batch_size, )
        if labels.ndim >= 2 and labels.shape[-1] == 1:
            labels = paddle.squeeze(labels, axis=[-1])

        loss = self._compute_loss(feats, labels, feats, labels)

        # compute contrastive loss from memory bank
        self.iter += 1
        if self.iter > self.start_iter:
            self.xbm.enqueue_dequeue(feats.detach(), labels.detach())
            xbm_feats, xbm_labels = self.xbm.get()
            xbm_loss = self._compute_loss(feats, labels, xbm_feats, xbm_labels)
            loss = loss + self.xbm_weight * xbm_loss

        return {'ContrastiveLoss_XBM': loss}
