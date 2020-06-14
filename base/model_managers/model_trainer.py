"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
import re
from abc import *
import torch
from torch import nn
from matplotlib import pyplot as plt

from base.model_managers.model_manager import ModelManager


class ModelTrainer(ModelManager):
    """
    트레이너들의 공통기능을 묶어서 만든 추상클래스입니다.

    1. EntityTrainer
    2. IntentClassifierTrainer
    3. InternRetrievalTrainer

    위와 같은 클래스들이 이 클래스를 상속받습니다.
    """

    train_data, test_data = None, None

    def train_model(self):
        errs, accs = [], []
        for i in range(self.epochs):
            err, acc = self._train_epoch()
            accs.append(acc)
            errs.append(err)

            self._print_log(i, err, acc)
            self._save_result('accuracy', accs)
            self._save_result('error', errs)

        self._draw_process('accuracy', 'red')
        self._draw_process('error', 'blue')
        print(self._store_and_test())

    @abstractmethod
    def _train_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _store_and_test(self):
        raise NotImplementedError

    def _load_dataset(self, data):
        self.train_data, self.test_data = data

    def _store_model(self, model, model_dir, model_file_path):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(model.state_dict(), model_file_path)

    def _print_log(self, step, train_err, train_acc):
        p = self.logging_precision
        print('step : {0} , train_error : {1} , train_acc : {2}'
              .format(step, round(train_err, p), round(train_acc, p)))

    def _draw_process(self, mode, color):
        f = open(self.root_dir + 'log/{}.txt'.format(mode), 'r')
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
        plt.savefig(self.root_dir + 'log/{}.png'.format(mode))
        plt.close()

    def _save_result(self, file_name, result):
        f = open(self.root_dir + '/log/{0}.txt'.format(file_name), 'w')
        f.write(str(result))
        f.close()

    @staticmethod
    def _initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)
