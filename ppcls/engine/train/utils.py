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

import paddle
import datetime
from ppcls.utils import logger
from ppcls.utils.misc import AverageMeter


def update_metric(trainer, out, batch, batch_size):
    # calc metric
    if trainer.train_metric_func is not None:
        metric_dict = trainer.train_metric_func(out, batch[-1])
        for key in metric_dict:
            if key not in trainer.output_info:
                trainer.output_info[key] = AverageMeter(key, '7.5f')
            trainer.output_info[key].update(
                float(metric_dict[key]), batch_size)


def update_loss(trainer, loss_dict, batch_size):
    # update_output_info
    for key in loss_dict:
        if key not in trainer.output_info:
            trainer.output_info[key] = AverageMeter(key, '7.5f')
        trainer.output_info[key].update(float(loss_dict[key]), batch_size)


def log_info(trainer, batch_size, epoch_id, iter_id):
    lr_msg = ", ".join([
        "lr({}): {:.8f}".format(type_name(lr), lr.get_lr())
        for i, lr in enumerate(trainer.lr_sch)
    ])
    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, trainer.output_info[key].avg)
        for key in trainer.output_info
    ])
    time_msg = "s, ".join([
        "{}: {:.5f}".format(key, trainer.time_info[key].avg)
        for key in trainer.time_info
    ])

    avg = trainer.time_info["batch_cost"].avg
    ips = batch_size / avg if avg != 0 else float("inf")
    ips_msg = "ips: {:.5f} samples/s".format(ips)

    global_epochs = trainer.config["Global"]["epochs"]
    eta_sec = (
        (trainer.config["Global"]["epochs"] - epoch_id + 1) *
        trainer.iter_per_epoch - iter_id) * trainer.time_info["batch_cost"].avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    max_mem_reserved_msg = ""
    max_mem_allocated_msg = ""
    max_mem_msg = ""
    print_mem_info = trainer.config["Global"].get("print_mem_info", False)
    if print_mem_info:
        if paddle.device.is_compiled_with_cuda():
            max_mem_reserved_msg = f"max_mem_reserved: {format(paddle.device.cuda.max_memory_reserved() / (1024 ** 2), '.2f')} MB"
            max_mem_allocated_msg = f"max_mem_allocated: {format(paddle.device.cuda.max_memory_allocated() / (1024 ** 2), '.2f')} MB"
            max_mem_msg = f", {max_mem_reserved_msg}, {max_mem_allocated_msg}"
    logger.info(
        f"[Train][Epoch {epoch_id}/{global_epochs}][Iter: {iter_id}/{trainer.iter_per_epoch}]{lr_msg}, {metric_msg}, {time_msg}, {ips_msg}, {eta_msg}{max_mem_msg}"
    )
    for key in trainer.time_info:
        trainer.time_info[key].reset()

    for i, lr in enumerate(trainer.lr_sch):
        logger.scaler(
            name="lr({})".format(type_name(lr)),
            value=lr.get_lr(),
            step=trainer.global_step,
            writer=trainer.vdl_writer)
    for key in trainer.output_info:
        logger.scaler(
            name="train_{}".format(key),
            value=trainer.output_info[key].avg,
            step=trainer.global_step,
            writer=trainer.vdl_writer)


def type_name(object: object) -> str:
    """get class name of an object"""
    return object.__class__.__name__
