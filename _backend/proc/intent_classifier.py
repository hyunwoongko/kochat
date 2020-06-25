"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
from time import time
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.optim import SGD

from _backend.decorators import intent
from _backend.proc.base.torch_processor import TorchProcessor
from _backend.proc.distance_estimator import DistanceEstimator
from _backend.proc.fallback_detector import FallbackDetector


@intent
class IntentClassifier(TorchProcessor):

    def __init__(self, model: nn.Module, loss: nn.Module, grid_search: bool = True):
        """
        Intent 분류 모델을 학습시키고 테스트 및 추론합니다.
        Intent Classifier는 Memory Network의 일종으로 거리 기반의 Fallback Detection을 수행할 수 있습니다.
        그러나 Nearest Neighbors 알고리즘을 Brute force가 아닌 KD-Tree 기반으로 구현했기 때문에 (Sklearn)
        샘플이 많아져도 빠르게 분류할 수 있습니다.

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

    def fit(self, dataset: tuple, test: bool = True):
        """
        Pytorch 모델을 학습/테스트하고 모델의 출력값을 다양한 방법으로 시각화합니다.
        최종적으로 학습된 모델을 저장합니다.
        IntentClassifier는 OOD 데이터셋이 존재하면 추가적으로 Fallback Detector도 학습시킵니다.

        :param dataset: 학습할 데이터셋
        :param test: 테스트 여부
        """

        super().fit(dataset, test)

        # ood 데이터가 있는 경우에 fallback detector 자동 학습/테스트
        if self.ood_train and self.ood_test:
            eta = time()
            self._ood_train_epoch()
            predicts, labels = self.__ood_test_epoch()

            self.metrics.evaluate(labels, predicts, mode='ood')
            report, _ = self.metrics.report(['in_dist', 'out_dist'], mode='ood')
            report = report.drop(columns=['macro avg'])

            self.visualizer.draw_report(report, mode='ood')
            self._print(name=self.fallback_detector.__class__.__name__,
                        msg='Training, ETA : {eta} sec'.format(eta=round(time() - eta, 4)))

    def predict(self, sequence: Tensor, calibrate: bool = False) -> str:
        """
        사용자의 입력에 inference합니다.
        OOD 데이터셋이 없는 경우 Fallback Threshold를 직접 수동으로
        맞춰야 하기 때문에 IntentClassifier는 Calibrate 모드를 지원합니다.
        calibrate를 True로 설정하면 현재 입력에 대한 샘플들의 거리를 알 수 있습니다.
        이 수치를 보고 Config의 fallback_detction_threshold를 조정해야합니다.

        :param sequence: 입력 시퀀스
        :param calibrate: Calibrate 모드 여부
        :return: 분류 결과 (클래스) 리턴
        """

        self._load_model()
        self.model.eval()

        _, feats = self._forward(sequence)
        predict, distance = self.distance_estimator.predict(sequence)

        if calibrate:
            self.__calibrate_msg(distance)

        if self.fallback_detction_criteria == 'auto':
            if self.fallback.predict(distance) == 0:
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

    def _train_epoch(self, epoch: int) -> tuple:
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

        if epoch % self.visualization_epoch == 0:
            self.visualizer.draw_feature_space(
                feats=feats,
                labels=labels,
                label_dict=self.label_dict,
                loss_name=self.loss.__class__.__name__,
                d_loss=self.d_loss,
                epoch=epoch,
                mode='train')

        return losses, predicts, labels

    def _test_epoch(self, epoch: int) -> tuple:
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

        if epoch % self.visualization_epoch == 0:
            self.visualizer.draw_feature_space(
                feats=feats,
                labels=labels,
                label_dict=self.label_dict,
                loss_name=self.loss.__class__.__name__,
                d_loss=self.d_loss,
                epoch=epoch,
                mode='test')

        return losses, predicts, labels

    def _ood_train_epoch(self):
        """
        out of distribution 데이터셋을 가지고
        Fallback Detector를 학습합니다.
        """

        feats_list, label_list = [], []
        self.model.eval()

        for (test, ood_train) in zip(self.test_data, self.ood_train):
            test_feats, test_labels, _ = test
            ood_train_feats, ood_train_labels, _, = ood_train

            feats = torch.cat([test_feats, ood_train_feats], dim=0).to(self.device)
            labels = torch.cat([test_labels, ood_train_labels], dim=0).to(self.device)
            _, feats = self._forward(feats)

            feats_list.append(feats)
            label_list.append(labels)

        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        distance, _ = self.distance_estimator.fit(feats, labels, mode='test')
        self.fallback_detector.fit(distance, labels, mode='train')

    def __ood_test_epoch(self) -> tuple:
        """
        out of distribution 데이터셋을 가지고
        Fallback Detector를 테스트합니다.
        """

        feats_list, label_list = [], []

        for feats, labels, lengths in self.ood_test:
            _, feats = self._forward(feats.to(self.device))

            feats_list.append(feats)
            label_list.append(labels)

        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        distance, _ = self.distance_estimator.fit(feats, labels, mode='test')
        predicts, labels = self.fallback_detector.fit(distance, labels, mode='test')
        return predicts, labels

    def _forward(self, feats: Tensor, labels: Tensor = None, lengths: Tensor = None) -> tuple:
        """
        모델의 feed forward에 대한 행동을 정의합니다.

        :param feats: 입력 feature
        :param labels: label 리스트
        :param lengths: 패딩을 제외한 입력의 길이 리스트
        :return: 모델의 출력(logits), features, loss
        """

        feats = self.model(feats)
        feats = self.model.ret_features(feats)
        logits = self.model.ret_logits(feats)

        if labels is None:
            return logits, feats

        loss = self.loss.compute_loss(labels, logits, feats)
        return logits, feats, loss

    def __calibrate_msg(self, distance: np.ndarray):
        print('\n=====================CALIBRATION_MODE=====================\n'
              '현재 입력하신 문장과 기존 문장들 사이의 거리 평균은 {0}이고\n'
              '가까운 샘플들과의 거리는 {1}입니다.\n'
              '이 수치를 보고 Config의 fallback_detction_threshold를 맞추세요.\n'
              'Fallback Detection은 거리평균/최솟값으로 설정할 수 있습니다.\n'
              .format(distance.mean(), distance[0][:5]))
