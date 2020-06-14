"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from base.model_managers.model_manager import Intent
from base.model_managers.model_trainer import ModelTrainer


class TrainClassifier(Intent, ModelTrainer):

    def __init__(self, model, dataset, label_dict):
        super().__init__()
        self.classes = len(label_dict)
        self.model = model.Model(vector_size=self.vector_size,
                                 max_len=self.max_len,
                                 d_model=self.d_model,
                                 layers=self.layers,
                                 classes=self.classes)

        self.model = self.model.cuda()
        self._initialize_weights(self.model)
        self._load_dataset(dataset)
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.intra_lr,
            weight_decay=self.weight_decay)

    def _train_epoch(self) -> tuple:
        errors, accuracies = [], []
        for train_feature, train_label in self.train_data:
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
            accuracies.append(self._get_accuracy(y, predict))

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    def _store_and_test(self) -> dict:
        self._store_model(self.model, self.intent_dir, self.intent_classifier_file)
        self.model.load_state_dict(torch.load(self.intent_classifier_file))
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().cuda()
        y = test_label.long().cuda()
        feature = self.model(x).float()
        classification = self.model.classifier(feature)

        _, predict = torch.max(classification, dim=1)
        return {'accuracy': self._get_accuracy(y, predict)}

    @staticmethod
    def _get_accuracy(predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all
