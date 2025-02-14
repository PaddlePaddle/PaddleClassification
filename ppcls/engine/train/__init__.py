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
from ppcls.engine.train.train import train_epoch
from ppcls.engine.train.train_fixmatch import train_epoch_fixmatch
from ppcls.engine.train.train_fixmatch_ccssl import train_epoch_fixmatch_ccssl
from ppcls.engine.train.train_progressive import train_epoch_progressive
from ppcls.engine.train.train_metabin import train_epoch_metabin
from ppcls.engine.train.train_multimodal import train_epoch_multimodal
