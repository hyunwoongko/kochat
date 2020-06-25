"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import Tensor
from torch import nn

from _backend.decorators import intent
from _backend.loss.cross_entropy_loss import CrossEntropyLoss
from _backend.proc.base.torch_processor import TorchProcessor


@intent
class SoftmaxClassifier(TorchProcessor):

    def __init__(self, model: nn.Module):
        """
        Intent 분류 모델을 학습시키고 테스트 및 추론합니다.
        Softmax Classification은 OOD 탐지기능이 없기 때문에 반드시 n개의 클래스 중 하나로 분류합니다.
        Calibrate되지 않은 Softmax score를 OOD 탐지의 기준으로 삼으면 발생할 수 있는 문제들은
        아래의 논문에 자세하게 설명되어있습니다. 때문에 OOD 탐지가 필요하면 IntentClassifier를 이용해주세요.

        - paper : Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images
        - arxiv : https://arxiv.org/abs/1412.1897

        :param model: Intent Classification 모델
        """

        self.label_dict = model.label_dict
        self.loss = CrossEntropyLoss(model.label_dict)
        super().__init__(model, model.parameters())

    def predict(self, sequence: Tensor) -> str:
        """
        사용자의 입력에 inference합니다.

        :param sequence: 입력 시퀀스
        :return: 분류 결과 (클래스) 리턴
        """

        self._load_model()
        self.model.eval()

        logits, _ = self._forward(sequence)
        _, predict = torch.max(logits, dim=1)
        return list(self.label_dict)[predict.item()]

    def _train_epoch(self, epoch: int) -> tuple:
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.
        
        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        loss_list, predict_list, label_list = [], [], []
        self.model.train()

        for feats, labels, lengths in self.train_data:
            feats, labels = feats.to(self.device), labels.to(self.device)
            predicts, losses = self._forward(feats, labels)
            losses = self._backward(losses)

            loss_list.append(losses)
            predict_list.append(predicts)
            label_list.append(labels)

        losses = sum(loss_list) / len(loss_list)
        predicts = torch.cat(predict_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        return losses, predicts, labels

    def _test_epoch(self, epoch: int) -> tuple:
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.
        
        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        loss_list, predict_list, label_list = [], [], []
        self.model.eval()

        for feats, labels, lengths in self.test_data:
            feats, labels = feats.to(self.device), labels.to(self.device)
            predicts, losses = self._forward(feats, labels)

            loss_list.append(losses)
            predict_list.append(predicts)
            label_list.append(labels)

        losses = sum(loss_list) / len(loss_list)
        predicts = torch.cat(predict_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        return losses, predicts, labels

    def _forward(self, feats: Tensor, labels: Tensor = None, lengths: Tensor = None):
        """
        모델의 feed forward에 대한 행동을 정의합니다.

        :param feats: 입력 feature
        :param labels: label 리스트
        :param lengths: 패딩을 제외한 입력의 길이 리스트
        :return: 모델의 예측, loss
        """

        feats = self.model(feats)
        logits = self.model.clf_logits(feats)
        _, predicts = torch.max(logits, dim=1)

        if labels is None:
            return predicts

        loss = self.loss.compute_loss(labels, logits, None)
        return predicts, loss
