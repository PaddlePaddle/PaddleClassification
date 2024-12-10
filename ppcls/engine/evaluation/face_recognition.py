# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import time
import platform
import paddle
import paddle.nn.functional as F

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger, all_gather


def face_recognition_eval(engine, epoch_id=0):
    # reset metric on beginning of eval
    if hasattr(engine.eval_metric_func, "reset"):
        engine.eval_metric_func.reset()
    output_info = dict()

    # log time_info for each batch
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)
    flip_test = engine.config["Global"].get("flip_test", False)
    feature_normalize = engine.config["Global"].get("feature_normalize", False)
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        time_info["reader_cost"].update(time.time() - tic)

        images_left, images_right, labels = [
            paddle.to_tensor(x) for x in batch[:3]]
        batch_remains = [paddle.to_tensor(x) for x in batch[3:]]
        labels = labels.astype('int64')
        batch_size = images_left.shape[0]

        # flip images
        if flip_test:
            images_left = paddle.concat(
                [images_left, paddle.flip(images_left, axis=-1)], 0)
            images_right = paddle.concat(
                [images_right, paddle.flip(images_right, axis=-1)], 0)

        with engine.auto_cast(is_eval=True):
            if engine.is_rec:
                out_left = engine.model(images_left, labels.reshape([-1, 1]))
                out_right = engine.model(images_right, labels.reshape([-1, 1]))
            else:
                out_left = engine.model(images_left)
                out_right = engine.model(images_right)

        # get features
        if engine.config["Global"].get("retrieval_feature_from",
                                      "features") == "features":
            # use output from neck as feature
            embeddings_left = out_left["features"]
            embeddings_right = out_right["features"]
        else:
            # use output from backbone as feature
            embeddings_left = out_left["backbone"]
            embeddings_right = out_right["backbone"]
        
        # normalize features
        if feature_normalize:
            embeddings_left = F.normalize(embeddings_left, p=2, axis=1)
            embeddings_right = F.normalize(embeddings_right, p=2, axis=1)
        
        # fuse features by sum up if flip_test is True
        if flip_test:
            embeddings_left = embeddings_left[:batch_size] + \
                              embeddings_left[batch_size:]
            embeddings_right = embeddings_right[:batch_size] + \
                               embeddings_right[batch_size:]

        # just for DistributedBatchSampler issue: repeat sampling
        current_samples = batch_size * paddle.distributed.get_world_size()
        accum_samples += current_samples

        # gather Tensor when distributed
        if paddle.distributed.get_world_size() > 1:
            embeddings_left = all_gather(embeddings_left)
            embeddings_right = all_gather(embeddings_right)
            labels = all_gather(labels)
            batch_remains = [all_gather(x) for x in batch_remains]

            # discard redundant padding sample(s) in the last batch
            if accum_samples > total_samples and not engine.use_dali:
                rest_num = total_samples + current_samples - accum_samples
                embeddings_left = embeddings_left[:rest_num]
                embeddings_right = embeddings_right[:rest_num]
                labels = labels[:rest_num]
                batch_remains = [x[:rest_num] for x in batch_remains]

        #  calc metric
        if engine.eval_metric_func is not None:
            engine.eval_metric_func(embeddings_left, embeddings_right, labels, 
                                    *batch_remains)
        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].val)
                    for key in output_info
            ])
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    if engine.use_dali:
        engine.eval_dataloader.reset()

    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, output_info[key].avg)
        for key in output_info
    ])
    metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    # do not try to save best eval.model
    if engine.eval_metric_func is None:
        return -1
    # return 1st metric in the dict
    return engine.eval_metric_func.avg

