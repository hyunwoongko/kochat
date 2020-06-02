"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn
from torch.optim import Adam

from config import Config
from util.dataset import Dataset
from util.graph_drawer import GraphDrawer


class SiameseTrainer:
    conf = Config()
    data = Dataset()

    def __init__(self, embed, model):
        self.embed = embed
        self.model = model.Model().to(self.conf.device)
        self.initialize_weights(self.model)
        self.margin = 1.0  # distance margin
        self.optimizer = Adam(
            lr=self.conf.intent_lr,
            params=self.model.parameters(),
            weight_decay=self.conf.intent_weight_decay)

    def train(self):
        errs, accs = [], []
        train_data, test_data = self.data.siamese_train(self.embed, self.conf.intent_datapath)
        for i in range(self.conf.intent_epochs):
            err = self.__train_epoch(self.model, train_data)
            errs.append(err)

            self.print_log(i, err)
            self.save_result('train_error', errs)

        drawer = GraphDrawer()
        drawer.draw('error', 'blue')
        torch.save(self.model.state_dict(),
                   self.conf.intent_storepath)

    def test(self):
        pass

    def __train_epoch(self, model, train_set):
        errors, accuracies = [], []
        for train_feature, train_label in train_set:
            self.optimizer.zero_grad()
            x1 = train_feature.float().cuda()[:, 0, :, :]
            x2 = train_feature.float().cuda()[:, 1, :, :]
            y = train_label.long().cuda()
            y1, y2 = model(siamese=True,
                           x1=x1.permute(0, 2, 1),
                           x2=x2.permute(0, 2, 1))

            dist_square = torch.sum((y2 - y1) ** 2, 1)
            dist = torch.sqrt(dist_square)
            mdist = self.margin - dist
            dist = torch.clamp(mdist, min=0.0)
            error = y * dist_square + (1 - y) * torch.pow(dist, 2)
            error = torch.sum(error) / 2.0 / y1.size()[0]
            error.backward()
            self.optimizer.step()

            errors.append(error.item())
        return sum(errors) / len(errors)

    def print_log(self, step, train_err):
        p = self.conf.intent_log_precision
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
