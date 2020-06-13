"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os

import torch
from torch import nn

from config import Config
from util.dataset import Dataset
from util.graph_drawer import GraphDrawer
from abc import *


class IntentTrainer:
    """
    Intent Retrieval Trainer와
    Intent Classifier Trainer의 부모 클래스입니다.
    """

    train_data, test_data = None, None

    def __init__(self, embed, model):
        self.conf = Config()
        self.data = Dataset()
        self.embed = embed
        self.__load_dataset(self.embed)
        self.model = model.Net().cuda()
        self.__initialize_weights(self.model)

    @abstractmethod
    def __train_epoch(self, train_set):
        """
        하위 클래스에서 반드시 구현해야함
        (템플릿 메소드 패턴)

        :param train_set: 트레이닝 데이터셋
        :return: 현재 에폭의 학습 결과
        """
        raise NotImplementedError

    @abstractmethod
    def __save_path(self) -> str:
        """
        하위 클래스에서 반드시 구현해야함
        (템플릿 메소드 패턴)

        :return: 학습할 경로 설정
        """
        raise NotImplementedError

    def train_model(self):
        errs, accs = [], []
        for i in range(self.conf.intent_epochs):
            err, acc = self.__train_epoch(self.train_data)
            accs.append(acc)
            errs.append(err)

            self.__print_log(i, err, acc)
            self.__save_result('accuracy', accs)
            self.__save_result('error', errs)

        drawer = GraphDrawer()
        drawer.draw('accuracy', 'red')
        drawer.draw('error', 'blue')

        if not os.path.exists(self.conf.intent_storepath):
            os.makedirs(self.conf.intent_storepath)

        torch.save(self.model.state_dict(), self.__save_path())

    def test_classification(self):
        self.model.load_state_dict(torch.load(self.__save_path()))
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().cuda()
        y = test_label.long().cuda()
        feature = self.model(x).float()
        classification = self.model.classifier(feature)

        _, predict = torch.max(classification, dim=1)
        acc = self.__get_accuracy(y, predict)
        print("classification test score is {}".format(acc))

    def __load_dataset(self, embed):
        self.train_data, self.test_data = \
            self.data.intent_train(embed)

    def __print_log(self, step, train_err, train_acc):
        p = self.conf.logging_precision
        print('step : {0} , train_error : {1} , train_acc : {2}'
              .format(step, round(train_err, p), round(train_acc, p)))

    def __save_result(self, file_name, result):
        f = open(self.conf.root_path + '/log/{0}.txt'.format(file_name), 'w')
        f.write(str(result))
        f.close()

    @staticmethod
    def __initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)

    @staticmethod
    def __get_accuracy(predict, label):
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all
