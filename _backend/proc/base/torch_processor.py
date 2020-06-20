"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
import re
from abc import abstractmethod, ABCMeta

import torch
from matplotlib import pyplot as plt
from torch import nn

from _backend.proc.base.base_processor import BaseProcessor


class TorchProcessor(BaseProcessor, metaclass=ABCMeta):
    def __init__(self, model):
        self.model = model.to(self.device)
        self.__initialize_weights(self.model)
        super().__init__(self.model)

    def fit(self, dataset, test=True):
        losses, accuracies = [], []
        self.train_data, self.test_data = dataset[0], dataset[1]
        self.model.train()

        for i in range(self.epochs + 1):
            loss, accuracy = self._fit(i)
            accuracies.append(accuracy)
            losses.append(loss)

            self.__print_log(i, loss, accuracy)
            self.__save_result('accuracy', accuracies)
            self.__save_result('loss', losses)

        self.__draw_accuracy_loss('accuracy', 'red')
        self.__draw_accuracy_loss('loss', 'blue')
        self._save_model()

        if test is True:
            self.test()

    @abstractmethod
    def _fit(self, epoch):
        raise NotImplementedError

    def _load_model(self):
        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.model_loaded = True
            self.model.load_state_dict(torch.load(self.model_file + '.pth'))

    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.model.state_dict(), self.model_file + '.pth')

    def __print_log(self, epoch, train_loss, train_accuracy):
        p = self.logging_precision
        print('{name} - epoch: {0}, train_loss: {1}, train_accuracy: {2}'
              .format(epoch, round(float(train_loss), p), round(float(train_accuracy), p),
                      name=self.__class__.__name__))

    def __save_result(self, mode, result):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        f = open(self.model_file + '_{}.txt'.format(mode), 'w')
        f.write(str(result))
        f.close()

    def __draw_accuracy_loss(self, mode, color):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        f = open(self.model_file + '_{}.txt'.format(mode), 'r')
        file = f.read()
        file = re.sub('\\[', '', file)
        file = re.sub('\\]', '', file)
        f.close()

        array = [float(i) for idx, i in enumerate(file.split(','))]
        plt.plot(array, color[0], label='train_{}'.format(mode))
        plt.xlabel('epochs')
        plt.ylabel(mode)
        plt.title('train ' + mode)
        plt.grid(True, which='both', axis='both')
        plt.savefig(self.model_file + '_{}.png'.format(mode))
        plt.close()

    def __initialize_weights(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)
