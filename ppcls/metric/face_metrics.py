# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from cmath import nan
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from ppcls.utils import logger


class FaceAccuracy(nn.Layer):
    """
    This code is modified from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval/verification.py
    """
    def __init__(self):
        super().__init__()
        self.embedding_left_list = []
        self.embedding_right_list = []
        self.label_list = []
        self.best_acc = 0.

    def forward(self, embeddings_left, embeddings_right, labels, *args):
        assert len(embeddings_left) == len(embeddings_right) == len(labels)
        self.embedding_left_list.append(normalize(embeddings_left.numpy()))
        self.embedding_right_list.append(normalize(embeddings_right.numpy()))
        self.label_list.append(labels.numpy())

        return {}
    
    def reset(self):
        self.embedding_left_list = []
        self.embedding_right_list = []
        self.label_list = []
        self.best_acc = 0.

    @property
    def avg(self):
        return self.best_acc

    @property
    def avg_info(self):
        embeddings_left = np.concatenate(self.embedding_left_list)
        embeddings_right = np.concatenate(self.embedding_right_list)
        labels = np.concatenate(self.label_list) 
        num_samples = len(embeddings_left)

        thresholds = np.arange(0, 4, 0.01)
        _, _, accuracy, best_thresholds = self.calculate_roc(thresholds, 
                                                             embeddings_left,
                                                             embeddings_right,
                                                             labels)
        self.best_acc = accuracy.mean()
        return "best_threshold: {:.4f}, acc: {:.4f}, num_samples: {}".format(
            best_thresholds.mean(), accuracy.mean(), num_samples)
        
    @staticmethod
    def calculate_roc(thresholds,
                      embeddings1,
                      embeddings2,
                      actual_issame,
                      nrof_folds=10,
                      pca=0):
        assert (embeddings1.shape[0] == embeddings2.shape[0])
        assert (embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        best_thresholds = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)
        # print('pca', pca)
        dist = None

        if pca == 0:
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff), 1)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if pca > 0:
                print('doing pca on', fold_idx)
                embed1_train = embeddings1[train_set]
                embed2_train = embeddings2[train_set]
                _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
                pca_model = PCA(n_components=pca)
                pca_model.fit(_embed_train)
                embed1 = pca_model.transform(embeddings1)
                embed2 = pca_model.transform(embeddings2)
                embed1 = normalize(embed1)
                embed2 = normalize(embed2)
                diff = np.subtract(embed1, embed2)
                dist = np.sum(np.square(diff), 1)

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx] = FaceAccuracy.calculate_accuracy(
                    threshold, dist[train_set], actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            best_thresholds[fold_idx] = thresholds[best_threshold_index]
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[
                    fold_idx, threshold_idx], _ = FaceAccuracy.calculate_accuracy(
                        threshold, dist[test_set], actual_issame[test_set])
            _, _, accuracy[fold_idx] = FaceAccuracy.calculate_accuracy(
                thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        return tpr, fpr, accuracy, best_thresholds


    @staticmethod
    def calculate_accuracy(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(
            np.logical_and(
                np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc


class FaceAccOnFiveDatasets(FaceAccuracy):
    dataname_to_idx = {
        "agedb_30": 0,
        "cfp_fp": 1,
        "lfw": 2,
        "cplfw": 3,
        "calfw": 4
    }
    idx_to_dataname = {v: k for k, v in dataname_to_idx.items()}

    def __init__(self):
        super().__init__()
        self.dataname_idx_list = []
    
    def forward(self, embeddings_left, embeddings_right, labels, 
                dataname_idxs, *args):
        assert len(embeddings_left) == len(dataname_idxs)
        dataname_idxs = dataname_idxs.astype('int64').numpy()
        self.dataname_idx_list.append(dataname_idxs)

        return super().forward(embeddings_left, embeddings_right, labels)
    
    def reset(self):
        super().reset()
        self.dataname_idx_list = []
    
    @property
    def avg_info(self):
        results = {}
        all_embeddings_left = np.concatenate(self.embedding_left_list)
        all_embeddings_right = np.concatenate(self.embedding_right_list)
        all_labels = np.concatenate(self.label_list)
        dataname_idxs = np.concatenate(self.dataname_idx_list)

        acc = []
        for dataname_idx in np.unique(dataname_idxs):
            dataname = self.idx_to_dataname[dataname_idx]
            mask = dataname_idxs == dataname_idx
            embeddings_left = all_embeddings_left[mask]
            embeddings_right = all_embeddings_right[mask]
            labels = all_labels[mask]

            thresholds = np.arange(0, 4, 0.01)
            _, _, accuracy, best_thresholds = self.calculate_roc(
                thresholds, embeddings_left, embeddings_right, labels)
            acc.append(accuracy.mean())
            results[f'{dataname}-best_threshold'] = f'{best_thresholds.mean():.4f}'
            results[f'{dataname}-acc'] = f'{accuracy.mean():.4f}'
            results[f'{dataname}-num_samples'] = f'{len(embeddings_left)}'
        self.best_acc = np.mean(acc)
        results['avg_acc'] = f'{self.best_acc:.4f}'

        info = ", ".join([f"{k}: {v}" for k, v in results.items()])
        return info
