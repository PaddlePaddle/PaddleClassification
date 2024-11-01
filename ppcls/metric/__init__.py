#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import copy
from collections import OrderedDict

from .avg_metrics import AvgMetrics
from .metrics import TopkAcc, mAP, mINP, Recallk, Precisionk
from .metrics import DistillationTopkAcc
from .metrics import GoogLeNetTopkAcc
from .metrics import HammingDistance, AccuracyScore
from .metrics import ATTRMetric
from .metrics import TprAtFpr, MultilabelMeanAccuracy
from .metrics import MultiLabelMAP
from .face_metrics import FaceAccuracy, FaceAccOnFiveDatasets


class CombinedMetrics(AvgMetrics):
    def __init__(self, config_list):
        super().__init__()
        self.metric_func_list = []
        assert isinstance(config_list, list), (
            'operator config should be a list')
        for config in config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            metric_name = list(config)[0]
            metric_params = config[metric_name]
            if metric_params is not None:
                self.metric_func_list.append(
                    eval(metric_name)(**metric_params))
            else:
                self.metric_func_list.append(eval(metric_name)())
        self.reset()

    def forward(self, *args, **kwargs):
        metric_dict = OrderedDict()
        for idx, metric_func in enumerate(self.metric_func_list):
            metric_dict.update(metric_func(*args, **kwargs))
        return metric_dict

    @property
    def avg_info(self):
        return ", ".join([metric.avg_info for metric in self.metric_func_list])

    @property
    def avg(self):
        return self.metric_func_list[0].avg

    def attr_res(self):
        return self.metric_func_list[0].attrmeter.res()

    def reset(self):
        for metric in self.metric_func_list:
            if hasattr(metric, "reset"):
                metric.reset()


def build_metrics(config):
    metrics_list = CombinedMetrics(copy.deepcopy(config))
    return metrics_list
