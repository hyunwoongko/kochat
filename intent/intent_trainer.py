"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import random

import torch
from torch import nn
from torch.optim import Adam

from configs import GlobalConfigs
from embedding.embedding import Embedding
from util.loader import TrainDataLoader
from util.tokenizer import Tokenizer


class IntentTrainer:
    conf = GlobalConfigs()
    data_loader = TrainDataLoader()
    emb = Embedding(conf.root_path + "models\\fasttext")

    def __init__(self, model, model_config, data_path, ratio=0.8):
        self.dataset = self.data_loader.load_intent(data_path=data_path)
        self.model_config = model_config
        self.model = model.to(self.conf.device)
        self.ratio = ratio
        self.optimizer = Adam(model.parameters(), lr=model_config.lr, weight_decay=model_config.weight_decay)
        self.loss = torch.nn.CrossEntropyLoss()

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
            data = self.emb.embed_single_row(data)
            data = self.pad_sequencing(data)
            embedded_train_dataset.append(data.unsqueeze(0))
            train_label.append(torch.tensor(label).unsqueeze(0))

        for data, label in test_dataset:
            data = self.emb.embed_single_row(data)
            data = self.pad_sequencing(data)
            embedded_test_dataset.append(data.unsqueeze(0))
            test_label.append(torch.tensor(label).unsqueeze(0))

        train_dataset = torch.cat(embedded_train_dataset, dim=0)
        test_dataset = torch.cat(embedded_test_dataset, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_label = torch.cat(test_label, dim=0)

        return train_dataset, train_label, test_dataset, test_label

    def __call__(self):
        train_dataset, train_label, test_dataset, test_label = self.preprocess()

        for i in range(self.model_config.epochs):
            train_err, train_acc = self.train(self.model, (train_dataset, train_label))
            test_err, test_acc = self.test(self.model, (test_dataset, test_label))
            print(train_err, test_err)

    def train(self, model, train_set):
        model.train()

        train_feature, train_label = train_set
        x = train_feature.float().cuda()
        y = train_label.long().cuda()
        y_ = model(x).float()

        self.optimizer.zero_grad()
        error = self.loss(y_, y)
        error.backward()
        self.optimizer.step()

        error = error.item()
        _, predict = torch.max(y_, dim=1)
        accuracy = self.get_accuracy(y, predict)
        return error, accuracy

    def test(self, model, test_set):
        model.eval()

        test_feature, test_label = test_set
        x = test_feature.float().cuda()
        y = test_label.long().cuda()
        y_ = model(x).float()

        error = self.loss(y_, y)
        error = error.item()
        _, predict = torch.max(y_, dim=1)
        accuracy = self.get_accuracy(y, predict)
        return error, accuracy

    def initialize_weights(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform(model.weight.data)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def get_accuracy(self, predict, label):
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all

    def save_result(self, file_name, result, step):
        f = open('./log/{0}_{1}.txt'.format(file_name, step), 'w')
        f.write(str(result))
        f.close()
