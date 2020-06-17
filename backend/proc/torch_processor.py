"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
import re
import torch
import numpy as np
import pandas as pd
from abc import abstractmethod, ABCMeta
from matplotlib import pyplot as plt
from torch import nn
from backend.proc.base_processor import BaseProcessor
from util.oop import override


class TorchProcessor(BaseProcessor, metaclass=ABCMeta):
    def __init__(self, model):
        self.model = model.to(self.device)
        self._initialize_weights(self.model)
        self.model_loaded = False
        super().__init__(self.model)

    @override(BaseProcessor)
    def train(self, dataset):
        losses, accuracies = [], []
        self.train_data, self.test_data = dataset
        for i in range(self.epochs):
            loss, accuracy = self._train_epoch()
            accuracies.append(accuracy)
            losses.append(loss)

            self._print_log(i, loss, accuracy)
            self._save_result('accuracy', accuracy)
            self._save_result('loss', loss)

        self._draw_accuracy_loss('accuracy', 'red')
        self._draw_accuracy_loss('loss', 'blue')
        self._save_model()

        test_result = str(self._test_epoch()) \
            .replace("'", "") \
            .replace("}", "") \
            .replace("{", "")

        print('{name} - {result}'
              .format(name=self.model.name, result=test_result))

    @override(BaseProcessor)
    def _load_model(self):
        self.model.eval()

        if not self.model_loaded:
            self.model_loaded = True
            self.model.load_state_dict(torch.load(self.model_file))

    @override(BaseProcessor)
    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.model.state_dict(), self.model_file)

    @abstractmethod
    def _train_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _test_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _get_accuracy(self, predict, label):
        raise NotImplementedError

    def _print_log(self, epoch, train_loss, train_accuracy):
        p = self.logging_precision
        print('{name} - epoch: {0}, train_loss: {1}, train_accuracy: {2}'
              .format(epoch, round(train_loss, p), round(train_accuracy, p),
                      name=self.model.name))

    def _save_result(self, mode, result):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        f = open(self.logs_dir + '{0}_{1}.png'.format(self.model.name, mode), 'w')
        f.write(str(result))
        f.close()

    def _draw_accuracy_loss(self, mode, color):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        f = open(self.logs_dir + '{0}_{1}.png'.format(self.model.name, mode), 'r')
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
        plt.savefig(self.logs_dir + '{0}_{1}.png'.format(self.model.name, mode))
        plt.close()

    def _initialize_weights(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)
