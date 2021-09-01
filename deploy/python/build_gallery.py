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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import cv2
import faiss
import numpy as np
from tqdm import tqdm
import pickle

from python.predict_rec import RecPredictor

from utils import logger
from utils import config


def split_datafile(data_file, image_root, delimiter="\t"):
    '''
        data_file: image path and info, which can be splitted by spacer
        image_root: image path root
        delimiter: delimiter
    '''
    gallery_images = []
    gallery_docs = []
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _, ori_line in enumerate(lines):
            line = ori_line.strip().split(delimiter)
            text_num = len(line)
            assert text_num >= 2, f"line({ori_line}) must be splitted into at least 2 parts, but got {text_num}"
            image_file = os.path.join(image_root, line[0])

            gallery_images.append(image_file)
            gallery_docs.append(ori_line.strip())

    return gallery_images, gallery_docs


class GalleryBuilder(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.build(config['IndexProcess'])

    def build(self, config):
        '''
            build index from scratch
        '''
        operation_method = config.get("index_operation", "new").lower()

        gallery_images, gallery_docs = split_datafile(
            config['data_file'], config['image_root'], config['delimiter'])
        if operation_method != "remove":
            gallery_features = self._extract_features(gallery_images, config)

        assert operation_method in [
            "new", "remove", "append"
        ], "Only append, remove and new operation are supported"
        if operation_method in ["remove", "append"]:
            assert os.path.join(
                config["index_dir"], "vector.index"
            ), "The vector.index dose not exist in {} when 'index_operation' is not None".format(
                config["index_dir"])
            assert os.path.join(
                config["index_dir"], "id_map.pkl"
            ), "The id_map.pkl dose not exist in {} when 'index_operation' is not None".format(
                config["index_dir"])
            index = faiss.read_index(
                os.path.join(config["index_dir"], "vector.index"))
            with open(os.path.join(config["index_dir"], "id_map.pkl"),
                      'rb') as fd:
                ids = pickle.load(fd)
            assert index.ntotal == len(ids.keys(
            )), "data number in index is not equal in in id_map"
        else:
            if not os.path.exists(config["index_dir"]):
                os.makedirs(config["index_dir"], exist_ok=True)
            index_method = config.get("index_method", "HNSW32")
            if index_method == "IVF":
                index_method = index_method + str(
                    min(int(len(gallery_images) // 8), 65536)) + ",Flat"
            dist_type = faiss.METRIC_INNER_PRODUCT if config[
                "dist_type"] == "IP" else faiss.METRIC_L2
            index = faiss.index_factory(config["embedding_size"], index_method,
                                        dist_type)
            index = faiss.IndexIDMap2(index)
            ids = {}

        if config["index_method"] == "HNSW32":
            logger.warning(
                "The HNSW32 method dose not support 'remove' operation")

        if operation_method != "remove":
            start_id = max(ids.keys()) + 1 if ids else 0
            ids_now = (
                np.arange(0, len(gallery_images)) + start_id).astype(np.int64)
            if operation_method == "new":
                index.train(gallery_features)
            index.add_with_ids(gallery_features, ids_now)

            for i, d in zip(list(ids_now), gallery_docs):
                ids[i] = d
        else:
            if config["index_method"] == "HNSW32":
                raise RuntimeError(
                    "The index_method: HNSW32 dose not support 'remove' operation"
                )
            remove_ids = list(
                filter(lambda k: ids.get(k) in gallery_docs, ids.keys()))
            remove_ids = np.asarray(remove_ids)
            index.remove_ids(remove_ids)
            for k in remove_ids:
                del ids[k]

        faiss.write_index(index,
                          os.path.join(config["index_dir"], "vector.index"))
        with open(os.path.join(config["index_dir"], "id_map.pkl"), 'wb') as fd:
            pickle.dump(ids, fd)

    def _extract_features(self, gallery_images, config):
        # extract gallery features
        gallery_features = np.zeros(
            [len(gallery_images), config['embedding_size']], dtype=np.float32)

        #construct batch imgs and do inference
        batch_size = config.get("batch_size", 32)
        batch_img = []
        for i, image_file in enumerate(tqdm(gallery_images)):
            img = cv2.imread(image_file)
            if img is None:
                logger.error("img empty, please check {}".format(image_file))
                exit()
            img = img[:, :, ::-1]
            batch_img.append(img)

            if (i + 1) % batch_size == 0:
                rec_feat = self.rec_predictor.predict(batch_img)
                gallery_features[i - batch_size + 1:i + 1, :] = rec_feat
                batch_img = []

        if len(batch_img) > 0:
            rec_feat = self.rec_predictor.predict(batch_img)
            gallery_features[-len(batch_img):, :] = rec_feat
            batch_img = []
        return gallery_features


def main(config):
    GalleryBuilder(config)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
