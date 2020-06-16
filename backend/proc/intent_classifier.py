"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from backend.decorators import intent
from backend.proc.torch_processpr import TorchProcessor
from util.oop import override


@intent
class IntentClassifier(TorchProcessor):
    train_data, test_data = None, None

    def __init__(self, model, label_dict):
        self.label_dict = label_dict
        super().__init__(model=model.Model(vector_size=self.vector_size,
                                           max_len=self.max_len,
                                           d_model=self.d_model,
                                           layers=self.layers,
                                           label_dict=self.label_dict))

        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.intra_lr,
            weight_decay=self.weight_decay)

    def inference(self, sequence):
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()

        output = self.model(sequence).float()
        output = self.model.classifier(output.squeeze())
        _, predict = torch.max(output, dim=0)
        return list(self.label_dict)[predict.item()]

    @override(TorchProcessor)
    def _train_epoch(self) -> tuple:
        self.model.train()

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

    @override(TorchProcessor)
    def _test_epoch(self) -> dict:
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().to(self.device)
        y = test_label.long().to(self.device)
        feature = self.model(x).to(self.device)
        classification = self.model.classifier(feature)

        _, predict = torch.max(classification, dim=1)
        return {'test_accuracy': self._get_accuracy(y, predict)}

    @override(TorchProcessor)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all
