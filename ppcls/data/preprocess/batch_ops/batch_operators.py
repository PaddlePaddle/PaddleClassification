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
from __future__ import unicode_literals
import random

import numpy as np

from ppcls.utils import logger
from ppcls.data.preprocess.ops.fmix import sample_mask


class BatchOperator(object):
    """ BatchOperator """

    def __init__(self, *args, **kwargs):
        pass

    def _unpack(self, batch):
        """ _unpack """
        assert isinstance(batch, list), \
                'batch should be a list filled with tuples (img, label)'
        bs = len(batch)
        assert bs > 0, 'size of the batch data should > 0'
        #imgs, labels = list(zip(*batch))
        imgs = []
        labels = []
        for item in batch:
            imgs.append(item[0])
            labels.append(item[1])
        return np.array(imgs), np.array(labels), bs

    def __call__(self, batch):
        return batch


class MixupOperator(BatchOperator):
    """ Mixup operator """

    def __init__(self, alpha: float=1.):
        """Build Mixup operator

        Args:
            alpha (float, optional): The parameter alpha of mixup. Defaults to 1..

        Raises:
            Exception: The value of parameter is illegal.
        """
        if alpha <= 0:
            raise Exception(
                f"Parameter \"alpha\" of Mixup should be greater than 0. \"alpha\": {alpha}."
            )
        self._alpha = alpha

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)
        lams = np.array([lam] * bs, dtype=np.float32)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], lams))


class CutmixOperator(BatchOperator):
    """ Cutmix operator """

    def __init__(self, alpha=0.2):
        """Build Cutmix operator

        Args:
            alpha (float, optional): The parameter alpha of cutmix. Defaults to 0.2.

        Raises:
            Exception: The value of parameter is illegal.
        """
        if alpha <= 0:
            raise Exception(
                f"Parameter \"alpha\" of Cutmix should be greater than 0. \"alpha\": {alpha}."
            )
        self._alpha = alpha

    def _rand_bbox(self, size, lam):
        """ _rand_bbox """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(imgs.shape, lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (float(bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.shape[-2] * imgs.shape[-1]))
        lams = np.array([lam] * bs, dtype=np.float32)
        return list(zip(imgs, labels, labels[idx], lams))


class FmixOperator(BatchOperator):
    """ Fmix operator """

    def __init__(self, alpha=1, decay_power=3, max_soft=0., reformulate=False):
        self._alpha = alpha
        self._decay_power = decay_power
        self._max_soft = max_soft
        self._reformulate = reformulate

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        size = (imgs.shape[2], imgs.shape[3])
        lam, mask = sample_mask(self._alpha, self._decay_power, \
                size, self._max_soft, self._reformulate)
        imgs = mask * imgs + (1 - mask) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], [lam] * bs))


class OpSampler(object):
    """ Sample a operator from  """

    def __init__(self, **op_dict):
        """Build OpSampler

        Raises:
            Exception: The parameter \"prob\" of operator(s) are be set error.
        """
        if len(op_dict) < 1:
            msg = f"ConfigWarning: No operator in \"OpSampler\". \"OpSampler\" has been skipped."

        self.ops = {}
        total_prob = 0
        for op_name in op_dict:
            param = op_dict[op_name]
            if "prob" not in param:
                msg = f"ConfigWarning: Parameter \"prob\" should be set when use operator in \"OpSampler\". The operator \"{op_name}\"'s prob has been set \"0\"."
                logger.warning(msg)
            prob = param.pop("prob", 0)
            total_prob += prob
            op = eval(op_name)(**param)
            self.ops[op] = prob

        if total_prob > 1:
            msg = f"ConfigError: The total prob of operators in \"OpSampler\" should be less 1."
            logger.error(msg)
            raise Exception(msg)

        # add "None Op" when total_prob < 1, "None Op" do nothing
        self.ops[None] = 1 - total_prob

    def __call__(self, batch):
        op = random.choices(
            list(self.ops.keys()), weights=list(self.ops.values()), k=1)[0]
        # return batch directly when None Op
        return op(batch) if op else batch
