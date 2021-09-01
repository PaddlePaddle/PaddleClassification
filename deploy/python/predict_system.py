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

import copy
import cv2
import numpy as np
import faiss
import pickle

from python.predict_rec import RecPredictor
from python.predict_det import DetPredictor

from utils import logger
from utils import config
from utils.get_image_list import get_image_list
from utils.draw_bbox import draw_bbox_results


class SystemPredictor(object):
    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)
        self.det_predictor = DetPredictor(config)

        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.return_k = self.config['IndexProcess']['return_k']

        index_dir = self.config["IndexProcess"]["index_dir"]
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "
        self.Searcher = faiss.read_index(
            os.path.join(index_dir, "vector.index"))
        with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)

    def append_self(self, results, shape):
        results.append({
            "class_id": 0,
            "score": 1.0,
            "bbox":
            np.array([0, 0, shape[1], shape[0]]),  # xmin, ymin, xmax, ymax
            "label_name": "foreground",
        })
        return results

    def nms_to_rec_results(self, results, thresh=0.1):
        filtered_results = []
        x1 = np.array([r["bbox"][0] for r in results]).astype("float32")
        y1 = np.array([r["bbox"][1] for r in results]).astype("float32")
        x2 = np.array([r["bbox"][2] for r in results]).astype("float32")
        y2 = np.array([r["bbox"][3] for r in results]).astype("float32")
        scores = np.array([r["rec_scores"] for r in results])

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
            filtered_results.append(results[i])

        return filtered_results

    def predict(self, img):
        output = []
        # st1: get all detection results
        results = self.det_predictor.predict(img)

        # st2: add the whole image for recognition to improve recall
        results = self.append_self(results, img.shape)

        # st3: recognition process, use score_thres to ensure accuracy
        for result in results:
            preds = {}
            xmin, ymin, xmax, ymax = result["bbox"].astype("int")
            crop_img = img[ymin:ymax, xmin:xmax, :].copy()
            rec_results = self.rec_predictor.predict(crop_img)
            preds["bbox"] = [xmin, ymin, xmax, ymax]
            scores, docs = self.Searcher.search(rec_results, self.return_k)
            # just top-1 result will be returned for the final
            if scores[0][0] >= self.config["IndexProcess"]["score_thres"]:
                preds["rec_docs"] = self.id_map[docs[0][0]].split()[1]
                preds["rec_scores"] = scores[0][0]
                output.append(preds)

        # st5: nms to the final results to avoid fetching duplicate results
        output = self.nms_to_rec_results(
            output, self.config["Global"]["rec_nms_thresold"])

        return output


def main(config):
    system_predictor = SystemPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        img = cv2.imread(image_file)[:, :, ::-1]
        output = system_predictor.predict(img)
        draw_bbox_results(img, output, image_file)
        print(output)
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
