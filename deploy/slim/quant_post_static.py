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

from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import paddle
import paddleslim
from paddle.jit import to_static
from paddleslim.analysis import dygraph_flops as flops

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))
from paddleslim.dygraph.quant import QAT

from ppcls.data import build_dataloader
from ppcls.utils import config as conf
from ppcls.utils.logger import init_logger


def main():
    args = conf.parse_args()
    config = conf.get_config(args.config, overrides=args.override, show=False)

    assert os.path.exists(
        os.path.join(config["Global"]["save_inference_dir"],
                     'inference.pdmodel')) and os.path.exists(
                         os.path.join(config["Global"]["save_inference_dir"],
                                      'inference.pdiparams'))
    config["DataLoader"]["Train"]["sampler"]["batch_size"] = 1
    config["DataLoader"]["Train"]["loader"]["num_workers"] = 0
    init_logger()
    device = paddle.set_device("cpu")
    train_dataloader = build_dataloader(config["DataLoader"], "Train", device,
                                        False)

    def sample_generator(loader):
        def __reader__():
            for indx, data in enumerate(loader):
                images = np.array(data[0])
                yield images

        return __reader__

    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir=config["Global"]["save_inference_dir"],
        model_filename='inference.pdmodel',
        params_filename='inference.pdiparams',
        quantize_model_path=os.path.join(
            config["Global"]["save_inference_dir"], "quant_post_static_model"),
        sample_generator=sample_generator(train_dataloader),
        batch_nums=10)


if __name__ == "__main__":
    main()
