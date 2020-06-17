"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from backend.decorators import intent
from backend.proc.torch_processor import TorchProcessor
from util.oop import override


@intent
class IntentClassifier(TorchProcessor):

    def __init__(self, model):
        super().__init__(model=model)
        self.label_dict = model.label_dict
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.model_lr,
            weight_decay=self.weight_decay)

    @override(TorchProcessor)
    def inference(self, sequence):
        self._load_model()
        logits = self.model(sequence).float()
        logits = self.model.clf_logits(logits.squeeze())
        _, predict = torch.max(logits, dim=0)
        return list(self.label_dict)[predict.item()]

    @override(TorchProcessor)
    def _train_epoch(self, epoch) -> tuple:
        self.model.train()

        losses, accuracies = [], []
        for train_feature, train_label in self.train_data:
            self.optimizer.zero_grad()
            x = train_feature.float().to(self.device)
            y = train_label.long().to(self.device)
            feats = self.model(x).float()
            logits = self.model.clf_logits(feats)

            loss = self.loss(logits, y)
            loss.backward()

            self.optimizer.step()
            losses.append(loss.item())
            _, predict = torch.max(logits, dim=1)
            acc = self._get_accuracy(y, predict)
            accuracies.append(acc)

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        return loss, accuracy

    @override(TorchProcessor)
    def _test_epoch(self) -> dict:
        self._load_model()
        test_feature, test_label = self.test_data
        x = test_feature.float().to(self.device)
        y = test_label.long().to(self.device)
        feats = self.model(x).to(self.device)
        logits = self.model.clf_logits(feats)

        _, predict = torch.max(logits, dim=1)
        return {'test_accuracy': self._get_accuracy(y, predict)}

    @override(TorchProcessor)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all
