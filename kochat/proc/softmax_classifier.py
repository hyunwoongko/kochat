"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import Tensor
from torch import nn

from kochat.decorators import intent
from kochat.loss.base_loss import BaseLoss
from kochat.proc.fallback_detector import FallbackDetector
from kochat.proc.intent_classifier import IntentClassifier


@intent
class SoftmaxClassifier(IntentClassifier):

    def __init__(self, model: nn.Module, loss: BaseLoss, grid_search: bool = True):
        """
        Distance Intent 분류 모델을 학습시키고 테스트 및 추론합니다.

        :param model: Intent Classification 모델
        :param loss: Loss 함수 종류
        """

        self.label_dict = model.label_dict
        self.loss = loss.to(self.device)
        self.fallback_detector = FallbackDetector(self.label_dict, grid_search)
        self.softmax = nn.Softmax(dim=1)
        super().__init__(model, model.parameters())

    def predict(self, sequence: Tensor, calibrate: bool = False) -> str:
        """
        사용자의 입력에 inference합니다.
        OOD 데이터셋이 없는 경우 Fallback Threshold를 직접 수동으로
        맞춰야 하기 때문에 IntentClassifier는 Calibrate 모드를 지원합니다.

        :param sequence: 입력 시퀀스
        :param calibrate: Calibrate 모드 여부
        :return: 분류 결과 (클래스) 리턴
        """

        self._load_model()
        self.model.eval()

        predict, softmax, _ = self._forward(sequence)
        softmax = softmax.unsqueeze(1)

        if calibrate:
            self._calibrate_msg(softmax)

        if self.softmax_fallback_detection_criteria == 'auto':
            if self.softmax_fallback_detection_threshold.predict(softmax) == 0:
                return list(self.label_dict)[predict[0]]

        else:
            if softmax.item() > self.fallback_detection_threshold:
                return list(self.label_dict)[predict[0]]

        return "FALLBACK"

    def _train_epoch(self, epoch: int) -> tuple:
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.

        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        loss_list, predict_list, feats_list, label_list = [], [], [], []
        self.model.train()

        for feats, labels, lengths in self.train_data:
            feats, labels = feats.to(self.device), labels.to(self.device)
            predicts, _, feats, losses = self._forward(feats, labels)
            losses = self._backward(losses)

            feats_list.append(feats)
            loss_list.append(losses)
            predict_list.append(predicts)
            label_list.append(labels)

        losses = sum(loss_list) / len(loss_list)
        feats = torch.cat(feats_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        if epoch % self.visualization_epoch == 0:
            self.visualizer.draw_feature_space(
                feats=feats,
                labels=labels,
                label_dict=self.label_dict,
                loss_name=self.loss.__class__.__name__,
                d_loss=self.d_loss,
                epoch=epoch,
                mode='train')

        return losses, labels, predicts

    def _test_epoch(self, epoch: int) -> tuple:
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.
        
        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        loss_list, predict_list, feats_list, label_list = [], [], [], []
        self.model.eval()

        for feats, labels, lengths in self.test_data:
            feats, labels = feats.to(self.device), labels.to(self.device)
            predicts, _, feats, losses = self._forward(feats, labels)

            feats_list.append(feats)
            loss_list.append(losses)
            predict_list.append(predicts)
            label_list.append(labels)

        losses = sum(loss_list) / len(loss_list)
        feats = torch.cat(feats_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        if epoch % self.visualization_epoch == 0:
            self.visualizer.draw_feature_space(
                feats=feats,
                labels=labels,
                label_dict=self.label_dict,
                loss_name=self.loss.__class__.__name__,
                d_loss=self.d_loss,
                epoch=epoch,
                mode='test')

        return losses, labels, predicts

    def _ood_train_epoch(self):
        """
        out of distribution 데이터셋을 가지고
        Fallback Detector를 학습합니다.
        """

        softmax_list, label_list = [], []
        self.model.eval()

        for (test, ood_train) in zip(self.test_data, self.ood_train):
            test_feats, test_labels, _ = test
            ood_train_feats, ood_train_labels, _, = ood_train

            feats = torch.cat([test_feats, ood_train_feats], dim=0).to(self.device)
            labels = torch.cat([test_labels, ood_train_labels], dim=0).to(self.device)
            _, softmax, _ = self._forward(feats)

            softmax_list.append(softmax)
            label_list.append(labels)

        softmax = torch.cat(softmax_list, dim=0).unsqueeze(1)
        labels = torch.cat(label_list, dim=0)
        self.fallback_detector.fit(softmax, labels, mode='train')

    def _ood_test_epoch(self) -> tuple:
        """
        out of distribution 데이터셋을 가지고
        Fallback Detector를 테스트합니다.
        """

        softmax_list, label_list = [], []

        for feats, labels, lengths in self.ood_test:
            feats, labels = feats.to(self.device), labels.to(self.device)
            _, softmax, _ = self._forward(feats)

            softmax_list.append(softmax)
            label_list.append(labels)

        softmax = torch.cat(softmax_list, dim=0).unsqueeze(1)
        labels = torch.cat(label_list, dim=0)

        predicts, labels = self.fallback_detector.fit(softmax, labels, mode='test')
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
        feats = self.model.features(feats)
        logits = self.model.classifier(feats)
        logits = self.softmax(logits)

        (softmax, predicts) = torch.max(logits, dim=1)

        if labels is None:
            return predicts, softmax, feats

        loss = self.loss.compute_loss(labels, logits, feats)
        return predicts, softmax, feats, loss

    def _calibrate_msg(self, logits: Tensor):
        print('\n=====================CALIBRATION_MODE=====================\n'
              '현재 입력하신 문장의 softmax logits은 {0}입니다.\n'
              '이 수치를 보고 Config의 fallback_detection_threshold를 맞추세요.\n'
              'SoftmaxClassifier는 cireteria가 "auto" 혹은 아무값으로 설정해주세요.\n'
              .format(logits.item()))
