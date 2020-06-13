"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from intent.abstract.intent_trainer import IntentTrainer


class IntentClassifierTrainer(IntentTrainer):

    def __init__(self, embed, model):
        super().__init__(embed, model)
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.conf.intent_intra_lr,
            weight_decay=self.conf.intent_weight_decay)

    def save_path(self) -> str:
        return self.conf.intent_clf_storefile

    def train_epoch(self, train_set):
        errors, accuracies = [], []
        for train_feature, train_label in train_set:
            x = train_feature.float().cuda()
            y = train_label.long().cuda()
            feature = self.model(x).float()
            classification = self.model.classifier(feature)

            error = self.loss(classification, y)
            self.loss.zero_grad()
            self.optimizer.zero_grad()
            error.backward()

            self.optimizer.step()
            errors.append(error.item())
            _, predict = torch.max(classification, dim=1)
            acc = self.get_accuracy(y, predict)
            accuracies.append(acc)

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    def test_classification(self):
        self.model.load_state_dict(torch.load(self.conf.intent_storefile))
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().cuda()
        y = test_label.long().cuda()
        feature = self.model(x).float()
        classification = self.model.classifier(feature)

        _, predict = torch.max(classification, dim=1)
        acc = self.get_accuracy(y, predict)
        print("INTENT : classification test accuracy is {}".format(acc))
