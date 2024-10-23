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

import os
import numpy as np
import pickle
from paddle.io import Dataset
from .common_dataset import create_operators
from ppcls.data.preprocess import transform as transform_func

class FaceEvalDataset(Dataset):
    def __init__(self,
                 dataset_root,
                 pair_label_path,
                 transform_ops=None,
                 delimiter=None):
        super().__init__()
        self._dataset_root = dataset_root
        self._pair_label_path = pair_label_path
        self.delimiter = delimiter if delimiter is not None else " "
        self._transform_ops = create_operators(transform_ops) if transform_ops \
            is not None else None

        self._load_anno()
    
    def _load_anno(self):
        assert os.path.exists(
            self._pair_label_path), f"pair label file {self._pair_label_path} does not exist"
        assert os.path.exists(
            self._dataset_root), f"path {self._dataset_root} does not exist."
        self.image_pairs = []
        self.labels = []

        with open(self._pair_label_path) as fd:
            lines = fd.readlines()
            for line in lines:
                line = line.strip().split(self.delimiter)

                left_img_path = os.path.join(self._dataset_root, line[0])
                assert os.path.exists(left_img_path), \
                    f"path {left_img_path} does not exist."
                right_img_path = os.path.join(self._dataset_root, line[1])
                assert os.path.exists(right_img_path), \
                    f"path {right_img_path} does not exist."
                self.image_pairs.append((left_img_path, right_img_path))

                label = np.int64(line[2])
                assert label in [0, 1], f"label must be 0 or 1, but got {label}"
                self.labels.append(label)

    def __getitem__(self, idx):
        with open(self.image_pairs[idx][0], 'rb') as f:
            img_left = f.read()
        with open(self.image_pairs[idx][1], 'rb') as f:
            img_right = f.read()
        if self._transform_ops is not None:
            img_left = transform_func(img_left, self._transform_ops)
            img_right = transform_func(img_right, self._transform_ops)
        
        img_left = img_left.transpose((2, 0, 1))
        img_right = img_right.transpose((2, 0, 1))
        return img_left, img_right, self.labels[idx]

    def __len__(self):
        return len(self.image_pairs)


class FiveFaceEvalDataset(Dataset):
    dataname_to_idx = {
        "agedb_30": 0,
        "cfp_fp": 1,
        "lfw": 2,
        "cplfw": 3,
        "calfw": 4
    }
    def __init__(self, 
                 val_data_path, 
                 val_targets=['agedb_30','cfp_fp','lfw'],
                 transform_ops=None):
        '''
        agedb_30: 0
        cfp_fp: 1
        lfw: 2
        cplfw: 3
        calfw: 4
        '''
        if isinstance(val_targets, str):
            val_targets = [val_targets]
        assert isinstance(val_targets, list)
        assert all([x in self.dataname_to_idx.keys() for x in val_targets]), \
            f"val_targets must be in {self.dataname_to_idx.keys()}"
        self._transform_ops = create_operators(transform_ops)

        # concat all dataset
        all_img_buffs = []
        all_issame = []
        all_dataname_idxs = []
        for dataname in val_targets:
            dataname_idx = self.dataname_to_idx[dataname]
            assert os.path.exists(
                os.path.join(val_data_path, dataname+".bin")), \
                f"{dataname}" f".bin not found in {val_data_path}"
            with open(os.path.join(val_data_path, dataname+".bin"), 'rb') as f:
                img_buffs, issame = pickle.load(f, encoding='bytes')
            for i in range(0, len(img_buffs), 2):
                left_buff, right_buff = img_buffs[i], img_buffs[i + 1]
                if isinstance(left_buff, np.ndarray):
                    left_buff = left_buff.tobytes()
                if isinstance(right_buff, np.ndarray):
                    right_buff = right_buff.tobytes()     
                all_img_buffs.append((left_buff, right_buff))
            all_issame.extend(list(issame))
            all_dataname_idxs.extend([dataname_idx] * len(issame))
            assert len(all_issame) == len(all_img_buffs)

        self.all_img_buffs = all_img_buffs
        self.all_issame = all_issame
        self.all_dataname_idxs = all_dataname_idxs

    def __getitem__(self, index):
        left_buff, right_buff = self.all_img_buffs[index]
        if self._transform_ops is not None:
            img_left = transform_func(left_buff, self._transform_ops)
            img_right = transform_func(right_buff, self._transform_ops)
        img_left = img_left.transpose((2, 0, 1))
        img_right = img_right.transpose((2, 0, 1))
        
        dataname_idx = self.all_dataname_idxs[index]
        label = np.int64(self.all_issame[index])
        return img_left, img_right, label, dataname_idx

    def __len__(self):
        return len(self.all_img_buffs)