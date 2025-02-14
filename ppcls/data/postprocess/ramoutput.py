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

import os
import numpy as np

import paddle
import paddle.nn.functional as F


class RamOutPut(object):
    def __init__(self,
                 language="cn",
                 tag_list="",
                 tag_list_chinese="",
                 threshold=0.68,
                 ram_class_threshold_path="",
                 delete_tag_index=[]):
        """
        """
        self.language = language
        assert tag_list, tag_list_chinese
        self.tag_list = self.load_tag_list(tag_list)
        self.delete_tag_index = delete_tag_index
        self.tag_list_chinese = self.load_tag_list(tag_list_chinese)
        self.num_class = len(self.tag_list)
        self.class_threshold = paddle.ones([self.num_class]) * threshold
        ram_class_threshold_path = ram_class_threshold_path
        with open(ram_class_threshold_path, "r", encoding="utf-8") as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key, value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, "r", encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    def __call__(self, logits, file_names=None):
        """
        logits is the result from model
        bs is the batch size from model
        file_names is useless but need in order to fit support framework of ppcls
        """
        bs = len(file_names)
        targets = paddle.where(
            F.sigmoid(logits) > self.class_threshold,
            paddle.to_tensor([1.0]), paddle.zeros(self.num_class))
        targets = targets.reshape([bs, -1])
        res = {}
        tag = targets.cpu().numpy()
        tag[:, self.delete_tag_index] = 0
        tag_output = []
        tag_output_chinese = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(" | ".join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(" | ".join(token_chinese))
        res["cn"] = tag_output_chinese
        res["en"] = tag_output
        res["all"] = f"en : {tag_output}, cn: {tag_output_chinese}"

        scores = F.sigmoid(logits).numpy().reshape([bs, -1])
        class_ids_list = []
        scores_list = []

        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            class_ids_list.append(index.tolist())
            scores_list.append(scores[b][index].tolist())



        outputformat = {
            "class_ids": class_ids_list,
            "scores": scores_list,
            "label_names": res[self.language]
        }

        return outputformat
