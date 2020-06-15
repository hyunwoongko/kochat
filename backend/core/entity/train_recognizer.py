"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from backend.base.base_manager import override
from backend.base.model_manager import Entity
from backend.base.model_trainer import ModelTrainer


class TrainRecognizer(Entity, ModelTrainer):

    def __init__(self, model, dataset, label_dict):
        super().__init__()
        self.classes = len(label_dict)
        self.model = model.Model(vector_size=self.vector_size,
                                 d_model=self.d_model,
                                 layers=self.layers,
                                 classes=self.classes,
                                 device=self.device)

        self.model = self.model.to(self.device)
        self.model.train()  # train 모드
        self._initialize_weights(self.model)
        self._load_dataset(dataset)
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)

    @override(ModelTrainer)
    def _train_epoch(self) -> tuple:
        errors, accuracies = [], []
        for train_feature, train_label in self.train_data:
            x = train_feature.float().cuda()
            y = train_label.long().cuda()
            out = self.model(x).float()

            error = self.loss(out, y)
            self.loss.zero_grad()
            self.optimizer.zero_grad()
            error.backward()

            self.optimizer.step()
            errors.append(error.item())
            _, predict = torch.max(out, dim=1)
            accuracies.append(self._get_accuracy(y, predict))

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    @override(ModelTrainer)
    def _store_and_test(self) -> dict:
        self._store_model(self.model, self.entity_dir, self.entity_recognizer_file)
        self.model.load_state_dict(torch.load(self.entity_recognizer_file))
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().cuda()
        y = test_label.long().cuda()
        out = self.model(x).float()

        _, predict = torch.max(out, dim=1)
        return {'accuracy': self._get_accuracy(y, predict)}

    @override(ModelTrainer)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            for j in zip(i[0], i[1]):
                all += 1
                if j[0] == j[1]:
                    correct += 1
        return correct / all
