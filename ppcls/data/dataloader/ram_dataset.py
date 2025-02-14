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

import json
import os
import re
import numpy as np

from paddle.io import Dataset
import paddle
from .common_dataset import create_operators
from ppcls.data.preprocess import transform
from ppcls.arch.clip.tokenizer import Tokenizer
from ppcls.arch.clip.clip import tokenize
from ppcls.arch.ram.ram import init_tokenizer

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(), )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption, )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


class RAMPretrainDataset(Dataset):
    def __init__(self,
                 ann_file,
                 width=384,
                 class_num=4585,
                 root='',
                 tag_list='',
                 model_name='',
                 transform_ops_ram=None,
                 transform_ops_clip=None):

        self.ann = []
        for f in ann_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann += ann
        self.width = width
        self.name = model_name.lower()
        self.transform_clip = create_operators(transform_ops_clip)
        self.transform = create_operators(transform_ops_ram)
        self.class_num = class_num
        self.root = root
        self.tag_list = self.load_tag_list(tag_list)
        self.bert_tokenizer = init_tokenizer()
        self.clip_tokenizer = Tokenizer()
        

    def __len__(self):
        return len(self.ann)
    
    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding='utf-8') as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list
    
    def collect_fn_list(self, data):
        image_ram_list = []
        text_list = []
        image_tag_list = []
        image_parse_tag_list = []
        image_clip_list = []
        for item in data:
            i1,i2,i3,i4,i5 = item
            image_ram_list.append(i1)
            text_list.append(i2)
            image_tag_list.append(i3)
            image_parse_tag_list.append(i4)
            image_clip_list.append(i5)
        image_rams = paddle.stack(image_ram_list)
        if self.name == "ram":
            text_list = self.bert_tokenizer(
                text_list,
                padding='longest',
                truncation=True,
                max_length=40,
                return_attention_mask=True,
                return_tensors='pd')
        else:
            text_list = tokenize(text_list, self.clip_tokenizer)
        image_tags = paddle.stack(image_tag_list)
        image_parse_tag_list = np.stack(image_parse_tag_list)
        tag_input = []
        for b in range(len(image_ram_list)):
            index = np.argwhere(image_parse_tag_list[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_input.append(' | '.join(token))
        image_parse_tags = self.bert_tokenizer(
            tag_input,
            padding='max_length',
            truncation=True,
            max_length=40,
            return_attention_mask=True,
            return_tensors='pd')
        image_clips = paddle.stack(image_clip_list)
        return [image_rams, text_list , image_tags, image_parse_tags, image_clips]

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path_use = os.path.join(self.root, ann['image_path'])

        try:
            image = Image.open(image_path_use).convert('RGB')
            image_ram = transform(image, self.transform)
            image_ram = paddle.to_tensor(image_ram)
            image224 = transform(image, self.transform_clip)
            image224 = paddle.to_tensor(image224)
        except:
            image224 = paddle.ones([3, 224, 224])
            image_ram = paddle.ones([3, self.width, self.width])

        num = ann['union_label_id']
        image_tag = np.zeros([self.class_num])
        image_tag[num] = 1
        image_tag = paddle.to_tensor(image_tag, dtype=paddle.int32)

        caption_index = np.random.randint(0, len(ann['caption']))

        caption = pre_caption(ann['caption'][caption_index], 30)

        num = ann['parse_label_id'][caption_index]
        parse_tag = np.zeros([self.class_num])
        parse_tag[num] = 1
        
        return [image_ram, caption, image_tag, parse_tag, image224]


