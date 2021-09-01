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

import paddle
from paddle import nn
import numpy as np
import paddleslim
from paddle.jit import to_static
from paddleslim.analysis import dygraph_flops as flops
import argparse
import paddle.distributed as dist
from visualdl import LogWriter

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))
from paddleslim.dygraph.quant import QAT

from ppcls.engine.trainer import Trainer
from ppcls.utils import config, logger
from ppcls.utils.logger import init_logger
from ppcls.utils.config import print_config
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from ppcls.data import build_dataloader
from ppcls.arch import apply_to_static
from ppcls.arch import build_model, RecModel, DistillationModel
from ppcls.arch.gears.identity_head import IdentityHead

quant_config = {
    # weight preprocess type, default is None and no preprocessing is performed. 
    'weight_preprocess_type': None,
    # activation preprocess type, default is None and no preprocessing is performed.
    'activation_preprocess_type': None,
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. default is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}


class ExportModel(nn.Layer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, config, model):
        super().__init__()
        self.base_model = model

        # we should choose a final model to export
        if isinstance(self.base_model, DistillationModel):
            self.infer_model_name = config["infer_model_name"]
        else:
            self.infer_model_name = None

        self.infer_output_key = config.get("infer_output_key", None)
        if self.infer_output_key == "features" and isinstance(self.base_model,
                                                              RecModel):
            self.base_model.head = IdentityHead()
        if config.get("infer_add_softmax", True):
            self.softmax = nn.Softmax(axis=-1)
        else:
            self.softmax = None

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def forward(self, x):
        x = self.base_model(x)
        if isinstance(x, list):
            x = x[0]
        if self.infer_model_name is not None:
            x = x[self.infer_model_name]
        if self.infer_output_key is not None:
            x = x[self.infer_output_key]
        if self.softmax is not None:
            x = self.softmax(x)
        return x


class Trainer_slim(Trainer):
    def __init__(self, config, mode="train"):

        self.mode = mode
        self.config = config
        self.output_dir = self.config['Global']['output_dir']

        log_file = os.path.join(self.output_dir, self.config["Arch"]["name"],
                                f"{mode}.log")
        init_logger(name='root', log_file=log_file)
        print_config(config)
        # set device
        assert self.config["Global"]["device"] in ["cpu", "gpu", "xpu"]
        self.device = paddle.set_device(self.config["Global"]["device"])
        # set dist
        self.config["Global"][
            "distributed"] = paddle.distributed.get_world_size() != 1

        if "Head" in self.config["Arch"]:
            self.is_rec = True
        else:
            self.is_rec = False

        self.model = build_model(self.config["Arch"])
        # set @to_static for benchmark, skip this by default.
        apply_to_static(self.config, self.model)

        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    self.model, self.config["Global"]["pretrained_model"])
            else:
                load_dygraph_pretrain(
                    self.model, self.config["Global"]["pretrained_model"])

        self.vdl_writer = None
        if self.config['Global']['use_visualdl'] and mode == "train":
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)
        logger.info('train with paddle {} and device {}'.format(
            paddle.__version__, self.device))
        # init members
        self.train_dataloader = None
        self.eval_dataloader = None
        self.gallery_dataloader = None
        self.query_dataloader = None
        self.eval_mode = self.config["Global"].get("eval_mode",
                                                   "classification")
        self.amp = True if "AMP" in self.config else False
        if self.amp and self.config["AMP"] is not None:
            self.scale_loss = self.config["AMP"].get("scale_loss", 1.0)
            self.use_dynamic_loss_scaling = self.config["AMP"].get(
                "use_dynamic_loss_scaling", False)
        else:
            self.scale_loss = 1.0
            self.use_dynamic_loss_scaling = False
        if self.amp:
            AMP_RELATED_FLAGS_SETTING = {
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
                'FLAGS_max_inplace_grad_add': 8,
            }
            paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)
        self.train_loss_func = None
        self.eval_loss_func = None
        self.train_metric_func = None
        self.eval_metric_func = None
        self.use_dali = self.config['Global'].get("use_dali", False)

        # for slim
        pact = self.config["Slim"].get("quant", False)
        self.pact = pact.get("name", False) if pact else pact

        if self.pact and str(self.pact.lower()) != 'pact':
            raise RuntimeError("The quantization only support 'PACT'!")
        if pact:
            quant_config["activation_preprocess_type"] = "PACT"
            self.quanter = QAT(config=quant_config)
            self.quanter.quantize(self.model)
            logger.info("QAT model summary:")
            paddle.summary(self.model, (1, 3, 224, 224))
        else:
            self.quanter = None

        prune_config = self.config["Slim"].get("prune", False)
        if prune_config:
            if prune_config["name"].lower() not in ["fpgm", "l1_norm"]:
                raise RuntimeError(
                    "The prune methods only support 'fpgm' and 'l1_norm'")
            else:
                logger.info("FLOPs before pruning: {}GFLOPs".format(
                    flops(self.model, [1] + self.config["Global"][
                        "image_shape"]) / 1e9))
                self.model.eval()

                if prune_config["name"].lower() == "fpgm":
                    self.model.eval()
                    self.pruner = paddleslim.dygraph.FPGMFilterPruner(
                        self.model, [1] + self.config["Global"]["image_shape"])
                else:
                    self.pruner = paddleslim.dygraph.L1NormFilterPruner(
                        self.model, [1] + self.config["Global"]["image_shape"])
                self.prune_model()
        else:
            self.pruner = None

        if self.quanter is None and self.pruner is None:
            logger.info("Training without slim")

        # for distributed training
        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()
            self.model = paddle.DataParallel(self.model)

    def export_inference_model(self):
        if os.path.exists(
                os.path.join(self.output_dir, self.config["Arch"]["name"],
                             "best_model.pdparams")):
            load_dygraph_pretrain(self.model,
                                  os.path.join(self.output_dir,
                                               self.config["Arch"]["name"],
                                               "best_model"))
        elif self.config["Global"].get(
                "pretrained_model", False) and os.path.exists(self.config[
                    "Global"]["pretrained_model"] + ".pdparams"):
            load_dygraph_pretrain(self.model,
                                  self.config["Global"]["pretrained_model"])
        else:
            raise RuntimeError(
                "The best_model or pretraine_model should exist to generate inference model"
            )
        model = ExportModel(self.config["Arch"], self.model)
        save_path = os.path.join(self.config["Global"]["save_inference_dir"],
                                 "inference")
        if self.quanter:
            self.quanter.save_quantized_model(
                model,
                save_path,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + config["Global"]["image_shape"],
                        dtype='float32')
                ])
        else:
            model = to_static(
                model,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.config["Global"]["image_shape"],
                        dtype='float32',
                        name="image")
                ])
            paddle.jit.save(model, save_path)

    def prune_model(self):
        params = []
        for sublayer in self.model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(sublayer, paddle.nn.Conv2D):
                    params.append(param.name)
        ratios = {}
        for param in params:
            ratios[param] = self.config["Slim"]["prune"]["pruned_ratio"]
        plan = self.pruner.prune_vars(ratios, [0])

        logger.info("FLOPs after pruning: {}GFLOPs; pruned ratio: {}".format(
            flops(self.model, [1] + self.config["Global"]["image_shape"]) /
            1e9, plan.pruned_flops))

        for param in self.model.parameters():
            if "conv2d" in param.name:
                logger.info("{}\t{}".format(param.name, param.shape))

        self.model.train()


def parse_args():
    parser = argparse.ArgumentParser(
        "generic-image-rec slim script, for train, eval and export inference model"
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'infer', 'export'],
        help='the different function')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    if args.mode == 'train':
        trainer = Trainer_slim(config, mode="train")
        trainer.train()
    elif args.mode == 'eval':
        trainer = Trainer_slim(config, mode="eval")
        trainer.eval()
    elif args.mode == 'infer':
        trainer = Trainer_slim(config, mode="infer")
        trainer.infer()
    else:
        trainer = Trainer_slim(config, mode="train")
        trainer.export_inference_model()
