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

import os
import sys
sys.path.insert(0, ".")

import time

import numpy as np
import paddle.nn as nn
from paddlehub.module.module import moduleinfo, serving

from hubserving.clas.params import get_default_confg
from python.predict_cls import ClsPredictor
from utils import config
from utils.encode_decode import b64_to_np


@moduleinfo(
    name="clas_system",
    version="1.0.0",
    summary="class system service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/class")
class ClasSystem(nn.Layer):
    def __init__(self, use_gpu=None, enable_mkldnn=None):
        """
        initialize with the necessary elements
        """
        self._config = self._load_config(
            use_gpu=use_gpu, enable_mkldnn=enable_mkldnn)
        self.cls_predictor = ClsPredictor(self._config)

    def _load_config(self, use_gpu=None, enable_mkldnn=None):
        cfg = get_default_confg()
        cfg = config.AttrDict(cfg)
        config.create_attr_dict(cfg)
        if use_gpu is not None:
            cfg.Global.use_gpu = use_gpu
        if enable_mkldnn is not None:
            cfg.Global.enable_mkldnn = enable_mkldnn
        cfg.enable_benchmark = False
        if cfg.Global.use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("Use GPU, GPU Memery:{}".format(cfg.Global.gpu_mem))
                print("CUDA_VISIBLE_DEVICES: ", _places)
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        else:
            print("Use CPU")
            print("Enable MKL-DNN") if enable_mkldnn else None
        return cfg

    def predict(self, inputs):
        if not isinstance(inputs, list):
            raise Exception(
                "The input data is inconsistent with expectations.")

        starttime = time.time()
        outputs = self.cls_predictor.predict(inputs)
        elapse = time.time() - starttime
        return {"prediction": outputs, "elapse": elapse}

    @serving
    def serving_method(self, images, revert_params):
        """
        Run as a service.
        """
        input_data = b64_to_np(images, revert_params)
        results = self.predict(inputs=list(input_data))
        return results


if __name__ == "__main__":
    import cv2
    import paddlehub as hub

    module = hub.Module(name="clas_system")
    img_path = "./hubserving/ILSVRC2012_val_00006666.JPEG"
    img = cv2.imread(img_path)[:, :, ::-1]
    img = cv2.resize(img, (224, 224)).transpose((2, 0, 1))
    res = module.predict([img.astype(np.float32)])
    print("The returned result of {}: {}".format(img_path, res))
