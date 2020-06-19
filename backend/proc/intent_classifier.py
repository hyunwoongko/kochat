"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from backend.decorators import intent
from backend.loss.softmax_loss import SoftmaxLoss
from backend.proc.base.torch_processor import TorchProcessor


@intent
class IntentClassifier(TorchProcessor):

    def __init__(self, model):
        super().__init__(model=model)
        self.label_dict = model.label_dict
        self.loss = SoftmaxLoss(model.label_dict)
        self.optimizers = [Adam(
            params=self.model.parameters(),
            lr=self.model_lr,
            weight_decay=self.weight_decay)]

        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizers[0],
            verbose=True,
            factor=self.lr_scheduler_factor,
            min_lr=self.lr_scheduler_min_lr,
            patience=self.lr_scheduler_patience)

    def predict(self, sequence):
        self._load_model()
        self.model.eval()

        logits = self.model(sequence).float()
        logits = self.model.clf_logits(logits.squeeze())
        _, predict = torch.max(logits, dim=0)
        return list(self.label_dict)[predict.item()]

    def _fit(self, epoch) -> tuple:
        loss_list, accuracy_list = [], []
        for train_feature, train_label, train_length in self.train_data:

            feats = train_feature.float().to(self.device)
            labels = train_label.long().to(self.device)
            feats = self.model(feats).float()
            logits = self.model.clf_logits(feats)

            total_loss = self.loss.compute_loss(labels, logits, None)
            self.loss.step(total_loss, self.optimizers)

            loss_list.append(total_loss.item())
            _, predict = torch.max(logits, dim=1)
            acc = self._get_accuracy(labels, predict)
            accuracy_list.append(acc)

        loss = sum(loss_list) / len(loss_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)

        if epoch > self.lr_scheduler_warm_up:
            self.lr_scheduler.step(loss)

        return loss, accuracy

    def test(self) -> dict:
        self._load_model()
        self.model.eval()

        test_feature, test_label, test_length = self.test_data
        feats = test_feature.float().to(self.device)
        labels = test_label.long().to(self.device)
        feats = self.model(feats).to(self.device)
        logits = self.model.clf_logits(feats)

        _, predict = torch.max(logits, dim=1)
        test_result = {'test_accuracy': self._get_accuracy(labels, predict)}

        print('{0} - {1}'.format(self.model.name, test_result))
        return test_result
