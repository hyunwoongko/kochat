"""
@auther Hyunwoong
@since 6/28/2020
@see https://github.com/gusdnd852
"""
from abc import ABCMeta, abstractmethod
from time import time
from typing import List

import torch
from torch import nn
from torch.nn import Parameter

from kochat.decorators import intent
from kochat.proc.torch_processor import TorchProcessor


@intent
class IntentClassifier(TorchProcessor, metaclass=ABCMeta):

    def __init__(self, model: nn.Module, parameters: Parameter or List[Parameter]):
        model = self.__add_classifier(model)
        super().__init__(model, parameters)

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
            predicts, labels = self._ood_test_epoch()

            self.metrics.evaluate(labels, predicts, mode='ood')
            report, _ = self.metrics.report(['in_dist', 'out_dist'], mode='ood')
            report = report.drop(columns=['macro avg'])

            self.visualizer.draw_report(report, mode='ood')
            self._print(name=self.fallback_detector.__class__.__name__,
                        msg='Training, ETA : {eta} sec'.format(eta=round(time() - eta, 4)))

    @abstractmethod
    def _ood_train_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _ood_test_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _calibrate_msg(self, *args):
        raise NotImplementedError

    def __add_classifier(self, model):
        sample = torch.randn(1, self.max_len, self.vector_size)
        sample = sample.to(self.device)
        output_size = model.to(self.device)(sample)

        features = nn.Linear(output_size.shape[1], self.d_loss)
        classifier = nn.Linear(self.d_loss, len(model.label_dict))
        setattr(model, 'features', features.to(self.device))
        setattr(model, 'classifier', classifier.to(self.device))
        return model
