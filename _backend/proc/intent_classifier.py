"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from _backend.decorators import intent
from _backend.proc.base.torch_processor import TorchProcessor
from _backend.proc.distance_estimator import DistanceEstimator
from _backend.proc.fallback_detector import FallbackDetector


@intent
class IntentClassifier(TorchProcessor):

    def __init__(self, model, loss, debug=True):
        super().__init__(model)
        self.label_dict = model.label_dict
        self.loss = loss.to(self.device)
        self.debug = debug

        self.optimizers = [Adam(
            params=self.model.parameters(),
            lr=self.model_lr,
            weight_decay=self.weight_decay)]

        if len(list(loss.parameters())) != 0:
            loss_opt = SGD(params=loss.parameters(), lr=self.loss_lr)
            self.optimizers.append(loss_opt)

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizers[0],
            verbose=True,
            factor=self.lr_scheduler_factor,
            min_lr=self.lr_scheduler_min_lr,
            patience=self.lr_scheduler_patience)

        self.features, self.ood_data, self.fallback = {}, None, None
        self.dist_estimator = DistanceEstimator(self.label_dict)

    def predict(self, dataset, calibrate=False):
        self._load_model()
        self.model.eval()

        feats = self.model(dataset).float()
        feats = self.model.ret_features(feats.squeeze())
        predict, distance = self.dist_estimator.inference(feats)

        if calibrate:
            self.__calibrate_msg(distance)

        if self.fallback_detction_criteria == 'auto':
            if self.fallback.inference(distance) == 0:
                return list(self.label_dict)[predict[0]]

        elif self.fallback_detction_criteria == 'mean':
            if distance.mean() < self.fallback_detction_threshold:
                return list(self.label_dict)[predict[0]]

        elif self.fallback_detction_criteria == 'min':
            if distance.min() < self.fallback_detction_threshold:
                return list(self.label_dict)[predict[0]]
        else:
            raise Exception("잘못된 dist_criteria입니다. [auto, mean, min]중 선택하세요")

        return "FALLBACK"

    def fit(self, dataset, test=True):
        self.train_data, self.test_data = dataset[0], dataset[1]
        super().fit((self.train_data, self.test_data), test=False)

        if len(dataset) > 2:
            # OOD 데이터 있으면 자동으로 Fallback Detector ON
            self.fallback = FallbackDetector()
            self.ood_data = (dataset[2], dataset[3])
            self.__train_fallback(self.ood_data)

        if test is True:
            self.test()

    def _fit(self, epoch) -> tuple:
        loss_list, accuracy_list, feat_list, label_list = [], [], [], []
        for train_feature, train_label, train_length in self.train_data:
            feats = train_feature.float().to(self.device)
            labels = train_label.long().to(self.device)
            feats = self.model.ret_features(self.model(feats))
            logits = self.model.ret_logits(feats)

            total_loss = self.loss.compute_loss(labels, logits, feats)
            self.loss.step(total_loss, self.optimizers)

            feat_list.append(feats)
            label_list.append(labels)
            loss_list.append(total_loss.item())

        loss = sum(loss_list) / len(loss_list)
        feat_list = torch.cat(feat_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        if epoch > self.lr_scheduler_warm_up:
            self.lr_scheduler.step(loss)

        if self.debug or self.epochs == epoch:
            self.dist_estimator.fit((feat_list, label_list))
            accuracy = self.dist_estimator.test((feat_list, label_list))
        else:
            accuracy = 0.0

        if epoch % self.visualization_epoch == 0:
            self.__draw_feature_space(feat_list, label_list, epoch)

        return loss, accuracy

    def test(self):
        self._load_model()
        self.model.eval()

        test_feature, test_label, test_length = self.test_data
        feats = test_feature.float().to(self.device)
        labels = test_label.long().to(self.device)
        feats = self.model.ret_features(self.model(feats))

        test_accuracy = self.dist_estimator.test((feats, labels))
        test_result = {"test_accuracy": test_accuracy}

        if self.ood_data is not None and self.fallback is not None:
            _, ood_test_data = self.ood_data
            ood_feats, ood_label, ood_length = ood_test_data
            ood_feats = ood_feats.to(self.device)
            ood_label = ood_label.to(self.device)
            ood_feats = self.model.ret_features(self.model(ood_feats))

            dist, label = self.dist_estimator.make_dist_dataset((ood_feats, ood_label))
            ood_result = self.fallback.test((dist, label))
            test_result['fallback_detection_test'] = ood_result

        print('{0} - {1}'.format(self.__class__.__name__, test_result))
        return test_result

    def __train_fallback(self, dataset):
        ood_train_data, _ = dataset
        test_feature, test_label, test_length = self.test_data
        ood_feature, ood_label, ood_length = ood_train_data
        feats = torch.cat([test_feature, ood_feature], dim=0).to(self.device)
        labels = torch.cat([test_label, ood_label], dim=0).to(self.device)

        feats = self.model.ret_features(self.model(feats))
        dist, label = self.dist_estimator.make_dist_dataset((feats, labels))
        self.fallback.fit((dist, label))

    def __draw_feature_space(self, feat, labels, epoch):
        feat = feat.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        if self.d_loss == 2:  # 2차원 시각화
            data = np.c_[feat, labels]
            data = pd.DataFrame(data, columns=['x', 'y', 'label'])
            ax = plt.figure().add_subplot()
            ax.scatter(data['x'], data['y'], marker='o', c=data['label'])

        else:  # 3차원 시각화
            if self.d_loss != 3:
                # 4차원 이상인 경우 PCA로 3차원으로 만듬
                inc_pca = IncrementalPCA(n_components=3)
                for batch_x in np.array_split(feat, self.batch_size):
                    inc_pca.partial_fit(batch_x)
                feat = inc_pca.transform(feat)

            data = np.c_[feat, labels]
            data = pd.DataFrame(data=data, columns=['x', 'y', 'z', 'label'])
            ax = plt.figure().gca(projection='3d')
            ax.scatter(data['x'], data['y'], data['z'], marker='o', c=data['label'])

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        plt.savefig(self.model_dir + '{0}_{1}D_{2}.png'
                    .format(self.loss.get_loss_name(), self.d_loss, epoch))

        plt.close()

    def __calibrate_msg(self, distance):
        print('\n=====================CALIBRATION_MODE=====================\n'
              '현재 입력하신 문장과 기존 문장들 사이의 거리 평균은 {0}이고\n'
              '가까운 샘플들과의 거리는 {1}입니다.\n'
              '이 수치를 보고 Config의 fallback_detction_threshold를 맞추세요.\n'
              'Fallback Detection은 거리평균/최솟값으로 설정할 수 있습니다.\n'
              .format(distance.mean(), distance[0][:5]))
