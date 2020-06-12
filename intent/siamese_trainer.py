"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn
from torch.optim import Adam
import pandas as pd

from config import Config
from intent.loss.constrastive_loss import ContrastiveLoss
from util.dataset import Dataset
from util.graph_drawer import GraphDrawer
from util.tokenizer import Tokenizer


class SiameseTrainer:
    conf = Config()
    data = Dataset()

    def __init__(self, embed, model):
        self.embed = embed
        self.model = model.Model().to(self.conf.device)
        self.initialize_weights(self.model)
        self.loss = ContrastiveLoss(margin=1.0)
        self.load_dataset(embed)
        self.optimizer = Adam(
            lr=self.conf.siamese_lr,
            params=self.model.parameters(),
            weight_decay=self.conf.siamese_weight_decay)

    def load_dataset(self, embed):
        self.train_data, self.test_data = \
            self.data.siamese_train(embed, self.conf.intent_datapath)

    def train(self):
        errs, accs = [], []
        for i in range(self.conf.siamese_epochs):
            err = self.__train_epoch(self.model, self.train_data)
            errs.append(err)

            self.print_log(i, err)
            self.save_result('train_error', errs)

        drawer = GraphDrawer()
        drawer.draw('error', 'blue')
        torch.save(self.model.state_dict(),
                   self.conf.siamese_storefile)

    def test(self):
        self.model.load_state_dict(torch.load(self.conf.intent_storefile))

    def __train_epoch(self, model, train_set):
        errors, accuracies = [], []
        for train_feature, train_label in train_set:
            self.optimizer.zero_grad()
            x1 = train_feature.float().cuda()[:, 0, :, :]
            x2 = train_feature.float().cuda()[:, 1, :, :]
            y = train_label.long().cuda()
            y1 = self.model(x1.permute(0, 2, 1)).float()
            y2 = self.model(x2.permute(0, 2, 1)).float()

            error = self.loss(y1, y2, y)
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
