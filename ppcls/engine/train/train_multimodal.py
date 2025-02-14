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

import time
import paddle
from ppcls.engine.train.utils import update_loss, update_metric, log_info, type_name
from ppcls.utils import profiler


def train_epoch_multimodal(engine, epoch_id, print_batch_step):
    tic = time.time()

    if not hasattr(engine, "train_dataloader_iter"):
        engine.train_dataloader_iter = iter(engine.train_dataloader)

    for iter_id in range(engine.iter_per_epoch):
        # fetch data batch from dataloader
        try:
            batch = next(engine.train_dataloader_iter)
        except Exception:
            # NOTE: reset DALI dataloader manually
            if engine.use_dali:
                engine.train_dataloader.reset()
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            batch = next(engine.train_dataloader_iter)
        assert isinstance(batch, tuple) or isinstance(batch, list)
        profiler.add_profiler_step(engine.config["profiler_options"])
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)

        batch_size = batch[0][0].shape[0]

        engine.global_step += 1

        # image input
        with engine.auto_cast(is_eval=False):
            out = forward(engine, batch)
            loss_dict = engine.train_loss_func(out, batch[1])

        # loss
        loss = loss_dict["loss"] / engine.update_freq

        # backward & step opt
        scaled = engine.scaler.scale(loss)
        scaled.backward()
        if (iter_id + 1) % engine.update_freq == 0:
            for i in range(len(engine.optimizer)):
                # optimizer.step() with auto amp
                engine.scaler.step(engine.optimizer[i])
                engine.scaler.update()

        if (iter_id + 1) % engine.update_freq == 0:
            # clear grad
            for i in range(len(engine.optimizer)):
                engine.optimizer[i].clear_grad()
            # step lr(by step)
            for i in range(len(engine.lr_sch)):
                if not getattr(engine.lr_sch[i], "by_epoch", False):
                    engine.lr_sch[i].step()
            # update ema
            if engine.ema:
                engine.model_ema.update(engine.model)

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)
        tic = time.time()

    # step lr(by epoch)
    for i in range(len(engine.lr_sch)):
        if getattr(engine.lr_sch[i], "by_epoch", False) and \
                type_name(engine.lr_sch[i]) != "ReduceOnPlateau":
            engine.lr_sch[i].step()


def forward(engine, batch):
    return engine.model(*batch)
