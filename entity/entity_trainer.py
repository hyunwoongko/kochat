import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from config import Config
from util.dataset import Dataset
from util.graph_drawer import GraphDrawer


class EntityTrainer:
    train_data, test_data = None, None

    def __init__(self, embed, model):
        self.conf = Config()
        self.data = Dataset()
        self.embed = embed
        self.load_dataset(embed)
        self.model = model.Net(len(self.data.label_list)).cuda()
        self.initialize_weights(self.model)
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.conf.entity_lr,
            weight_decay=self.conf.entity_weight_decay)

    def load_dataset(self, embed):
        self.train_data, self.test_data = \
            self.data.entity_train(embed)

    def train(self):
        errs, accs = [], []
        self.load_dataset(self.embed)

        for i in range(self.conf.entity_epochs):
            err, acc = self.__train_epoch(self.model, self.train_data)
            accs.append(acc)
            errs.append(err)

            self.print_log(i, err, acc)
            self.save_result('train_accuracy', accs)
            self.save_result('train_error', errs)

        drawer = GraphDrawer()
        drawer.draw('accuracy', 'red')
        drawer.draw('error', 'blue')

        if not os.path.exists(self.conf.entity_storepath):
            os.makedirs(self.conf.entity_storepath)

        torch.save(self.model.state_dict(), self.conf.entity_storefile)

    def test_classification(self):
        print("INTENT : test start ...")
        self.model.load_state_dict(torch.load(self.conf.intent_storefile))
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().cuda()
        y = test_label.long().cuda()
        y_ = self.model(x.permute(0, 2, 1)).float()
        y_ = y_.permute(1, 2, 0)

        _, predict = y_.max(dim=1)
        acc = self.get_accuracy(y, predict)
        print("INTENT : test accuracy is {}".format(acc))

    def __train_epoch(self, model, train_set):
        errors, accuracies = [], []
        for train_feature, train_label in train_set:
            x = train_feature.float().cuda()
            y = train_label.long().cuda()
            y_ = model(x.permute(0, 2, 1)).float()
            y_ = y_.permute(1, 2, 0)

            self.optimizer.zero_grad()
            error = self.loss(y_, y)
            error.backward()
            self.optimizer.step()

            errors.append(error.item())
            _, predict = y_.max(dim=1)
            accuracies.append(self.get_accuracy(y, predict))

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    def print_log(self, step, train_err, train_acc):
        p = self.conf.logging_precision
        print('step : {0} , train_error : {1} , train_acc : {2}'
              .format(step, round(train_err, p), round(train_acc, p)))

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
            for j in zip(i[0], i[1]):
                all += 1
                if j[0] == j[1]:
                    correct += 1
        return correct / all
