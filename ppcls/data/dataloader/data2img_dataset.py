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

import numpy as np
import os
from os import path as osp

import paddle
from paddle.io import Dataset

from .common_dataset import create_operators
from ppcls.data.preprocess import transform
from ppcls.arch.clip.tokenizer import Tokenizer
from ppcls.arch.clip.clip import tokenize

from PIL import Image
from PIL import ImageFile

"""
this dataset is used to load the data following the formate of Img2Dataset
https://github.com/rom1504/img2dataset
the strcture is:
{dataset}/train/{index_id}.jpg image information
{dataset}/train/{index_id}.txt text information following prompts format such as a photo of {label}, a view of {lable}
{dataset}/train/{index_id}.json meta information such as negative prompts 
"""
class Img2Dataset(Dataset):
    def __init__(self, root_path, split="train", transform=None):
        super().__init__()

        self.root = osp.join(root_path,split)
        assert osp.exists(self.root)

        self.images = []
        self.texts = []
        self.metas = []
        self.init()
        self.text_tokenizer = Tokenizer()
        self.transform = create_operators(transform)
        
    
    def __len__(self):
        return len(self.images)
    
    def collect_fn_list(self, data):
        img_list = []
        text_list = []
        for item in data:
            img, text = item
            img_list.append(img)
            text_list.append(text)
        
        imgs = paddle.stack(img_list)
        tokens = tokenize(text_list, self.text_tokenizer)
        return imgs, tokens
    
    def load_image(self, path):

        image = Image.open(path).convert('RGB')
        return transform(image,self.transform) if self.transform else image
    
    def load_text(self, path):
        prompt = ""
        with open(path, "r") as f:
            prompt = f.readlines()[0]
        
        return prompt
    
    def __getitem__(self, idx):
        
        img, text = self.images[idx], self.texts[idx]
        img_tensor = paddle.to_tensor(self.load_image(img))
        text = self.load_text(text)
        return [img_tensor, text]

        

    def init(self):
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith("jpg") or file.endswith("png"):
                    self.images.append(os.path.join(root,file))
                elif file.endswith("txt"):
                    self.texts.append(os.path.join(root,file))
                elif file.endswith("json"):
                    self.metas.append(os.path.join(root,file))
        
