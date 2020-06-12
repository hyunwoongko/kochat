"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os

import torch
from torch import nn
from torch.optim import Adam

from config import Config
from intent.loss.center_loss import CenterLoss
from util.dataset import Dataset
from util.graph_drawer import GraphDrawer


class CenterTrainer:
    train_data, test_data = None, None
    conf = Config()
    data = Dataset()

    def __init__(self, embed, model):
        self.embed = embed
        self.model = model.Model().to(self.conf.device)
        self.initialize_weights(self.model)
        self.center_loss = CenterLoss()
        self.optimizer = Adam(
            lr=self.conf.siamese_lr,
            params=self.model.parameters(),
            weight_decay=self.conf.siamese_weight_decay)

    def load_dataset(self, embed):
        self.train_data, self.test_data = \
            self.data.intent_train(embed, self.conf.intent_datapath)

    def train(self):
        errs, accs = [], []
        print("INTENT : LOAD DATASET...")
        self.load_dataset(self.embed)

        print("INTENT : START TRAIN !")
        for i in range(self.conf.center_epochs):
            err = self.__train_epoch(self.model, self.train_data)
            errs.append(err)

            self.print_log(i, err)
            self.save_result('error', errs)

        drawer = GraphDrawer()
        drawer.draw('error', 'blue')

        if not os.path.exists(self.conf.intent_storepath):
            os.makedirs(self.conf.center_storefile)

        torch.save(self.model.state_dict(), self.conf.center_storefile)

    def test(self):
        print("test start ...")

    def __train_epoch(self, model, train_set):
        errors, accuracies = [], []
        for train_feature, train_label in train_set:
            x = train_feature.float().cuda()
            y = train_label.long().cuda()
            y_ = model(x.permute(0, 2, 1)).float()

            self.optimizer.zero_grad()
            error = self.center_loss(y_, y)
            error.backward()
            self.optimizer.step()
            errors.append(error.item())

        return sum(errors) / len(errors)

    def print_log(self, step, train_err):
        p = self.conf.siamese_log_precision
        print('step : {0} , train_error : {1}'
              .format(step, round(train_err, p)))

    def save_result(self, file_name, result):
        f = open(self.conf.root_path + '/log/{0}.txt'.format(file_name), 'w')
        f.write(str(result))
        f.close()

    @staticmethod
    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)

    @staticmethod
    def get_accuracy(predict, label):
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all
