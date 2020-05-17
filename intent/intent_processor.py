"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import random

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from configs import GlobalConfigs
from util.dataset import TrainDataLoader
from util.graph_drawer import GraphDrawer


class IntentProcessor:
    conf = GlobalConfigs()
    data_loader = TrainDataLoader()

    def __init__(self, emb, model, store_path, data_path):
        self.emb = emb
        self.store_path = store_path
        self.ratio = self.conf.intent_ratio

        self.model = model.Model().to(self.conf.device)
        self.dataset = self.data_loader.load_intent(data_path=data_path)
        self.loss = self.conf.intent_loss
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.conf.intent_lr,
                              weight_decay=self.conf.intent_weight_decay)

    def pad_sequencing(self, sequence):
        if sequence.size()[0] > self.conf.max_len:
            sequence = sequence[:self.conf.max_len]
        else:
            pad = torch.zeros(self.conf.max_len, self.conf.vector_size)
            for i in range(sequence.size()[0]):
                pad[i] = sequence[i]
            sequence = pad

        return sequence

    def preprocess(self):
        data, label = self.dataset['data'], self.dataset['label']
        self.dataset = [zipped for zipped in zip(data, label)]
        random.shuffle(self.dataset)

        split_point = int(len(self.dataset) * self.ratio)
        train_dataset = self.dataset[:split_point]
        test_dataset = self.dataset[split_point:]

        embedded_train_dataset, embedded_test_dataset = [], []
        train_label, test_label = [], []
        for data, label in train_dataset:
            data = self.emb.embed(data)
            data = self.pad_sequencing(data)
            embedded_train_dataset.append(data.unsqueeze(0))
            train_label.append(torch.tensor(label).unsqueeze(0))

        for data, label in test_dataset:
            data = self.emb.embed(data)
            data = self.pad_sequencing(data)
            embedded_test_dataset.append(data.unsqueeze(0))
            test_label.append(torch.tensor(label).unsqueeze(0))

        train_dataset = torch.cat(embedded_train_dataset, dim=0)
        test_dataset = torch.cat(embedded_test_dataset, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_label = torch.cat(test_label, dim=0)

        train_set = TensorDataset(train_dataset, train_label)
        train_set = DataLoader(train_set, batch_size=self.conf.batch_size, shuffle=True)
        test_set = TensorDataset(test_dataset, test_label)
        test_set = DataLoader(test_set, batch_size=self.conf.batch_size, shuffle=True)

        return train_set, test_set

    def train(self):
        train_dataset, test_dataset = self.preprocess()
        train_errors, train_accuracies = [], []
        test_errors, test_accuracies = [], []

        for i in range(self.conf.intent_epochs):
            train_err, train_acc = self.__train_epoch(self.model, train_dataset)
            test_err, test_acc = self.__test_epoch(self.model, test_dataset)

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
            error = self.loss(y_, y)
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
            error = self.loss(y_, y)
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
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def get_accuracy(predict, label):
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all
