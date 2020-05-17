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


class IntentClassifier:
    conf = Config()
    data = Dataset()

    def __init__(self, embed, model):
        self.embed = embed
        self.model = model.Model().to(self.conf.device)
        self.initialize_weights(self.model)
        self.optimizer = Adam(
            lr=self.conf.intent_lr,
            params=self.model.parameters(),
            weight_decay=self.conf.intent_weight_decay)

    def train(self):
        train_data, test_data = self.data.intent_train(self.embed, self.conf.intent_datapath)
        train_errors, train_accuracies, test_errors, test_accuracies = [], [], [], []

        for i in range(self.conf.intent_epochs):
            train_err, train_acc = self.__train_epoch(self.model, train_data)
            test_err, test_acc = self.__test_epoch(self.model, test_data)

            train_accuracies.append(train_acc)
            train_errors.append(train_err)
            test_accuracies.append(test_acc)
            test_errors.append(test_err)

            self.print_log(i, train_err, test_err, train_acc, test_acc)
            self.save_result('train_accuracy', train_accuracies)
            self.save_result('train_error', train_errors)
            self.save_result('test_accuracy', test_accuracies)
            self.save_result('test_error', test_errors)

        drawer = GraphDrawer()
        drawer.draw_both()

    def __train_epoch(self, model, train_set):
        model.train()
        errors, accuracies = [], []
        for train_feature, train_label in train_set:
            x = train_feature.float().cuda()
            y = train_label.long().cuda()
            y_ = model.forward_once(x.permute(0, 2, 1)).float()

            self.optimizer.zero_grad()
            error = self.conf.intent_loss(y_, y)
            error.backward()
            self.optimizer.step()

            errors.append(error.item())
            _, predict = torch.max(y_, dim=1)
            accuracies.append(self.get_accuracy(y, predict))

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    def __test_epoch(self, model, test_set):
        model.eval()
        errors, accuracies = [], []
        for test_feature, test_label in test_set:
            x = test_feature.float().cuda()
            y = test_label.long().cuda()
            y_ = model.forward_once(x.permute(0, 2, 1)).float()

            self.optimizer.zero_grad()
            error = self.conf.intent_loss(y_, y)
            error.backward()
            self.optimizer.step()

            errors.append(error.item())
            _, predict = torch.max(y_, dim=1)
            accuracies.append(self.get_accuracy(y, predict))

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    def print_log(self, step, train_err, test_err, train_acc, test_acc):
        print('step : {0} , train_error : {1} , test_error : {2}, train_acc : {3}, test_acc : {4}'
              .format(step,
                      round(train_err, self.conf.intent_log_precision),
                      round(test_err, self.conf.intent_log_precision),
                      round(train_acc, self.conf.intent_log_precision),
                      round(test_acc, self.conf.intent_log_precision)))

    def save_result(self, file_name, result):
        f = open(self.conf.root_path + '\\log\\{0}.txt'.format(file_name), 'w')
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
