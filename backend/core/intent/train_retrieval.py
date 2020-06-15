"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
from collections import Counter

import torch
from torch.optim import Adam, SGD

from backend.base.base_manager import override
from backend.base.model_manager import Intent
from backend.base.model_trainer import ModelTrainer
from backend.core.loss.center_loss import CenterLoss
from backend.core.loss.lsoftmax_loss import LargeMarginSoftmaxLoss


class TrainRetrieval(Intent, ModelTrainer):
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
        self.model.train()  # train 모드
        self._initialize_weights(self.model)
        self._load_dataset(dataset)
        self.intra_class_loss = LargeMarginSoftmaxLoss(self.intra_factor)
        self.inter_class_loss = CenterLoss(self.inter_factor, self.d_model, self.classes)

        self.parameter_set = list(self.model.parameters()) + list(self.inter_class_loss.parameters())
        self.inter_class_optimizer = SGD(params=self.inter_class_loss.parameters(), lr=self.inter_lr)
        self.intra_class_optimizer = Adam(params=self.parameter_set, lr=self.intra_lr,
                                          weight_decay=self.weight_decay)

    @override(ModelTrainer)
    def _train_epoch(self) -> tuple:
        errors, accuracies = [], []
        for train_feature, train_label in self.train_data:
            x = train_feature.float().to(self.device)
            y = train_label.long().to(self.device)
            feature = self.model(x).float()
            classification = self.model.classifier(feature)

            error = self.intra_class_loss(classification, y)
            error += self.inter_class_loss(feature, y)
            self.intra_class_optimizer.zero_grad()
            self.inter_class_optimizer.zero_grad()
            error.backward()

            # Center Loss의 Center 위치 옮겨주기
            for param in self.inter_class_loss.parameters():
                param.grad.data *= (self.inter_lr
                                    / (self.inter_factor
                                       * self.inter_lr))

            self.intra_class_optimizer.step()
            self.inter_class_optimizer.step()
            errors.append(error.item())
            _, predict = torch.max(classification, dim=1)
            accuracies.append(self._get_accuracy(y, predict))

        error = sum(errors) / len(errors)
        accuracy = sum(accuracies) / len(accuracies)
        return error, accuracy

    @override(ModelTrainer)
    def _store_and_test(self) -> dict:
        self._store_model(self.model, self.intent_dir, self.intent_classifier_file)
        self.model.load_state_dict(torch.load(self.intent_classifier_file))
        self.model.eval()
        result = {'classification_result': self._test_classification(),
                  'in_distribution_result': self._test_in_distribution()}

        return result

    @override(ModelTrainer)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all

    def _test_classification(self) -> float:
        test_feature, test_label = self.test_data
        x = test_feature.float().to(self.device)
        y = test_label.long().to(self.device)
        feature = self.model(x).float()
        classification = self.model.classifier(feature)

        _, predict = torch.max(classification, dim=1)
        print(predict)
        return self._get_accuracy(y, predict)

    def _test_in_distribution(self) -> float:
        predict, label, train_neighbors = [], [], []
        for train_feature, train_label in self.train_data:
            for sample in zip(train_feature, train_label):
                x, y = sample[0].to(self.device), sample[1].to(self.device)
                y_ = self.model(x.unsqueeze(0))
                train_neighbors.append((y_, y))

        test_feature, test_label = self.test_data
        for i, sample in enumerate(zip(test_feature, test_label)):
            x, y = sample[0].to(self.device), sample[1].to(self.device)
            label.append(y)
            y_ = self.model(x.unsqueeze(0))
            nearest_neighbors = []

            for neighbor in train_neighbors:
                _x, _y = neighbor[0], neighbor[1]
                dist = ((y_ - _x) ** 2) * 1000
                nearest_neighbors.append((dist.mean().item(), _y.item()))

            nearest_neighbors = sorted(nearest_neighbors, key=lambda z: z[0])
            nearest_neighbors = [n[1] for n in nearest_neighbors[:10]]  # 10개만
            out = Counter(nearest_neighbors).most_common()[0][0]
            predict.append(out)
            proceed = (i / len(test_feature)) * 100

            print("in distribution test : {}% done"
                  .format(round(proceed, self.logging_precision)))

        self.model.load_state_dict(torch.load(self.intent_retrieval_file))
        return self._get_accuracy(predict, label)
