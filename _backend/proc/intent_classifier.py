"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.optim import SGD

from _backend.decorators import intent
from _backend.proc.base.torch_processor import TorchProcessor
from _backend.proc.distance_estimator import DistanceEstimator
from _backend.proc.fallback_detector import FallbackDetector


@intent
class IntentClassifier(TorchProcessor):

    def __init__(self, model, loss, grid_search=True):
        """
        Intent 분류 모델을 학습시키고 테스트 및 추론합니다.
        Intent Classifier는 Memory Network의 일종으로 거리 기반의 Fallback Detection을 수행할 수 있습니다.
        그러나 Nearest Neighbors 알고리즘을 Brute force가 아닌 KD-Tree 기반으로 구현했기 때문에 (Sklearn)
        샘플이 많아져도 Classification 속도가 빠릅니다.

        :param model: Intent Classification 모델
        :param loss: Loss 함수 종류
        """

        self.label_dict = model.label_dict
        self.loss = loss.to(self.device)
        self.distance_estimator = DistanceEstimator(grid_search)
        self.fallback_detector = FallbackDetector(self.label_dict, grid_search)
        super().__init__(model, model.parameters())

        if len(list(loss.parameters())) != 0:
            loss_opt = SGD(params=loss.parameters(), lr=self.loss_lr)
            self.optimizers.append(loss_opt)

    def predict(self, sequence, calibrate=False):
        pass

    def _train_epoch(self, epoch):
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.

        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        loss_list, feats_list, label_list = [], [], []
        self.model.train()

        for feats, labels, lengths in self.train_data:
            feats, labels = feats.to(self.device), labels.to(self.device)
            logits, feats, losses = self._forward(feats, labels)
            losses = self._backward(losses)

            loss_list.append(losses)
            feats_list.append(feats)
            label_list.append(labels)

        losses = sum(loss_list) / len(loss_list)
        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        predicts, distance = \
            self.distance_estimator.fit(feats, labels, mode='train')
        return losses, predicts, labels

    def _test_epoch(self, epoch):
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.

        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        loss_list, feats_list, label_list = [], [], []
        self.model.eval()

        for feats, labels, lengths in self.test_data:
            feats, labels = feats.to(self.device), labels.to(self.device)
            logits, feats, losses = self._forward(feats, labels)

            loss_list.append(losses)
            feats_list.append(feats)
            label_list.append(labels)

        losses = sum(loss_list) / len(loss_list)
        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        predicts, distance = \
            self.distance_estimator.fit(feats, labels, mode='test')
        return losses, predicts, labels

    def _forward(self, feats, labels=None, lengths=None):
        """
        모델의 feed forward에 대한 행동을 정의합니다.

        :param feats: 입력 feature
        :param labels: label 리스트
        :return: 모델의 예측, loss
        """

        feats = self.model(feats)
        feats = self.model.ret_features(feats)
        logits = self.model.ret_logits(feats)

        if labels is None:
            return logits, feats

        loss = self.loss.compute_loss(labels, logits, feats)
        return logits, feats, loss
