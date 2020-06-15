"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from backend.base.base_manager import override
from backend.base.model_manager import Intent
from backend.base.model_trainer import ModelTrainer


class TrainClassifier(Intent, ModelTrainer):
    train_data, test_data = None, None

    def __init__(self, model, dataset, label_dict):
        super().__init__()
        self.classes = len(label_dict)
        self.model = model.Model(vector_size=self.vector_size,
                                 max_len=self.max_len,
                                 d_model=self.d_model,
                                 layers=self.layers,
                                 classes=self.classes)

        self.model = self.model.to(self.device)
        self.model.train() # train 모드
        self._initialize_weights(self.model)
        self._load_dataset(dataset)
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.intra_lr,
            weight_decay=self.weight_decay)

    @override(ModelTrainer)
    def _train_epoch(self) -> tuple:
        errors, accuracies = [], []
        for train_feature, train_label in self.train_data:
            self.optimizer.zero_grad()
            x = train_feature.float().to(self.device)
            y = train_label.long().to(self.device)
            feature = self.model(x).float()
            classification = self.model.classifier(feature)

            error = self.loss(classification, y)
            error.backward()

            self.optimizer.step()
            errors.append(error.item())
            _, predict = torch.max(classification, dim=1)
            acc = self._get_accuracy(y, predict)
            accuracies.append(acc)

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    @override(ModelTrainer)
    def _store_and_test(self) -> dict:
        self._store_model(self.model, self.intent_dir, self.intent_classifier_file)
        self.model.load_state_dict(torch.load(self.intent_classifier_file))
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().to(self.device)
        y = test_label.long().to(self.device)
        feature = self.model(x).to(self.device)
        classification = self.model.classifier(feature)

        _, predict = torch.max(classification, dim=1)
        return {'test_accuracy': self._get_accuracy(y, predict)}

    @override(ModelTrainer)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all
