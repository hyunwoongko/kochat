"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from abc import abstractmethod
from time import time
from typing import List

import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kochat.proc.base_processor import BaseProcessor
from kochat.utils.metrics import Metrics
from kochat.utils.visualizer import Visualizer


class TorchProcessor(BaseProcessor):

    def __init__(self, model: nn.Module, parameters: Parameter or List[Parameter]):
        """
        Pytorch 모델의 Training, Testing, Inference
        등을 관장하는 프로세서 클래스입니다.

        :param model: Pytorch 모델을 입력해야합니다.
        """

        super().__init__(model)
        self.visualizer = Visualizer(self.model_dir, self.model_file)
        self.metrics = Metrics(self.logging_precision)
        self.model = model.to(self.device)
        self.__initialize_weights(self.model)

        # Model Optimizer로 Adam Optimizer 사용
        self.optimizers = [Adam(
            params=parameters,
            lr=self.model_lr,
            weight_decay=self.weight_decay)]

        # ReduceLROnPlateau Scheduler 사용
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizers[0],
            verbose=True,
            factor=self.lr_scheduler_factor,
            min_lr=self.lr_scheduler_min_lr,
            patience=self.lr_scheduler_patience)

    def fit(self, dataset: tuple, test: bool = True):
        """
        Pytorch 모델을 학습/테스트하고
        모델의 출력값을 다양한 방법으로 시각화합니다.
        최종적으로 학습된 모델을 저장합니다.

        :param dataset: 학습할 데이터셋
        :param test: 테스트 여부
        """

        # 데이터 셋 unpacking
        self.train_data = dataset[0]
        self.test_data = dataset[1]

        if len(dataset) > 2:
            self.ood_train = dataset[2]
            self.ood_test = dataset[3]

        for i in range(self.epochs + 1):
            eta = time()
            loss, label, predict = self._train_epoch(i)
            self.__visualize(loss, label, predict, mode='train')
            # training epoch + visualization

            if test:
                loss, label, predict = self._test_epoch(i)
                self.__visualize(loss, label, predict, mode='test')
                # testing epoch + visualization

            if i > self.lr_scheduler_warm_up:
                self.lr_scheduler.step(loss)

            if i % self.save_epoch == 0:
                self._save_model()

            self._print('Epoch : {epoch}, ETA : {eta} sec '
                        .format(epoch=i, eta=round(time() - eta, 4)))

    @abstractmethod
    def _train_epoch(self, epoch: int):
        raise NotImplementedError

    @abstractmethod
    def _test_epoch(self, epoch: int):
        raise NotImplementedError

    def _load_model(self):
        """
        저장된 모델을 불러옵니다.
        """

        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.model_loaded = True
            self.model.load_state_dict(torch.load(self.model_file + '.pth'))

    def _save_model(self):
        """
        모델을 저장장치에 저장합니다.
        """

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.model.state_dict(), self.model_file + '.pth')

    def __initialize_weights(self, model: nn.Module):
        """
        model의 가중치를 초기화합니다.
        기본값으로 He Initalization을 사용합니다.

        :param model: 초기화할 모델
        """

        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)

    def __visualize(self, loss: Tensor, label: Tensor, predict: Tensor, mode: str):
        """
        모델의 feed forward 결과를 다양한 방법으로 시각화합니다.

        :param loss: 해당 에폭의 loss
        :param label: 데이터셋의 label
        :param predict: 모델의 predict
        :param mode: train or test
        """

        # 결과 계산하고 저장함
        eval_dict = self.metrics.evaluate(label, predict, mode=mode)
        report, matrix = self.metrics.report(self.label_dict, mode)
        self.visualizer.save_result(loss, eval_dict, mode=mode)

        # 결과를 시각화하여 출력함
        self.visualizer.draw_matrix(matrix, list(self.label_dict), mode)
        self.visualizer.draw_report(report, mode=mode)
        self.visualizer.draw_graphs()

    @abstractmethod
    def _forward(self, feats: Tensor, labels: Tensor = None, lengths: Tensor = None):
        raise NotImplementedError

    def _backward(self, loss: Tensor):
        """
        모든 trainable parameter에 대한 
        backpropation을 진행합니다.

        :param loss: backprop 이전 loss
        :return: backprop 이후 loss
        """

        for opt in self.optimizers: opt.zero_grad()
        loss.backward()
        for opt in self.optimizers: opt.step()
        return loss
