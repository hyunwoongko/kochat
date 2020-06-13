"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from collections import Counter

import torch
from torch import nn
from torch.optim import Adam, SGD

from config import Config
from intent.abstract.intent_trainer import IntentTrainer
from intent.loss.margin_softmax_loss import MarginSoftmaxLoss
from intent.loss.center_loss import CenterLoss
from util.dataset import Dataset
from util.graph_drawer import GraphDrawer


class IntentRetrievalTrainer(IntentTrainer):
    train_data, test_data = None, None

    def __init__(self, embed, model):
        super().__init__(embed, model)
        self.intra_class_loss, self.inter_class_loss = MarginSoftmaxLoss(), CenterLoss()
        self.parameter_set = list(self.model.parameters()) + list(self.inter_class_loss.parameters())
        self.inter_class_optimizer = SGD(params=self.inter_class_loss.parameters(), lr=self.conf.intent_inter_lr)
        self.intra_class_optimizer = Adam(params=self.parameter_set, lr=self.conf.intent_intra_lr,
                                          weight_decay=self.conf.intent_weight_decay)

    def __train_epoch(self, train_set):
        errors, accuracies = [], []
        for train_feature, train_label in train_set:

            x = train_feature.float().cuda()
            y = train_label.long().cuda()
            feature = self.model(x).float()
            classification = self.model.classifier(feature)

            error = self.intra_class_loss(classification, y)
            error += self.inter_class_loss(feature, y)
            self.intra_class_optimizer.zero_grad()
            self.inter_class_optimizer.zero_grad()
            error.backward()

            # Center Loss의 Center 위치 옮겨주기
            for param in self.inter_class_loss.parameters():
                param.grad.data *= (self.conf.intent_inter_lr
                                    / (self.inter_class_loss.reg_gamma
                                       * self.conf.intent_inter_lr))

            self.intra_class_optimizer.step()
            self.inter_class_optimizer.step()
            errors.append(error.item())
            _, predict = torch.max(classification, dim=1)
            acc = self.get_accuracy(y, predict)
            accuracies.append(acc)

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    def test_in_distribution(self):
        self.model.load_state_dict(torch.load(self.conf.intent_storefile))
        self.model.eval()

        train_neighbors = []
        for train_feature, train_label in self.train_data:
            for sample in zip(train_feature, train_label):
                x, y = sample[0].cuda(), sample[1].cuda()
                y_ = self.model(x.unsqueeze(0))
                train_neighbors.append((y_, y))

        test_feature, test_label = self.test_data
        all, correct = 0, 0
        for i, sample in enumerate(zip(test_feature, test_label)):
            x, y = sample[0].cuda(), sample[1].cuda()
            y_ = self.model(x.unsqueeze(0))
            nearest_neighbors = []

            for neighbor in train_neighbors:
                _x, _y = neighbor[0], neighbor[1]
                dist = ((y_ - _x) ** 2) * 1000
                nearest_neighbors.append((dist.mean().item(), _y.item()))

            nearest_neighbors = sorted(nearest_neighbors, key=lambda z: z[0])
            nearest_neighbors = [n[1] for n in nearest_neighbors[:10]]  # 10개만
            out = Counter(nearest_neighbors).most_common()[0][0]

            all += 1
            if y == out:
                correct += 1

            proceed = (i / len(test_feature)) * 100
            print(round(proceed, self.conf.logging_precision), '% done.')

        print('In Distribution Test Score : ', (correct / all) * 100, '%')
