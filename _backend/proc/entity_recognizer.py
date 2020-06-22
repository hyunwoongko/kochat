"""
@author : Hyunwoong
@when : 6/20/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from _backend.decorators import entity
from _backend.loss.crf_loss import CRFLoss
from _backend.loss.base.masking import Masking
from _backend.proc.base.torch_processor import TorchProcessor


@entity
class EntityRecognizer(TorchProcessor):

    def __init__(self, model, loss, masking=True):
        super().__init__(model)
        self.label_dict = model.label_dict
        self.loss = loss.to(self.device)
        self.masking = Masking() if masking else None
        parameter = list(model.parameters())

        if len(list(loss.parameters())) != 0:
            parameter += list(loss.parameters())

        self.optimizers = [Adam(
            params=parameter,
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

        length = self.__get_length(sequence)
        output = self.model(sequence).float()
        output = output.squeeze().t()
        _, predict = torch.max(output, dim=1)
        output = [list(self.label_dict.keys())[i.item()] for i in predict]
        return output[:length]

    def _fit(self, epoch) -> tuple:
        loss_list, accuracy_list = [], []
        for train_feature, train_label, train_length in self.train_data:
            feats = train_feature.float().to(self.device)
            labels = train_label.long().to(self.device)
            logits = self.model(feats).float()

            mask = self.masking(train_length) if self.masking else None
            total_loss = self.loss.compute_loss(labels, logits, feats, mask)
            for opt in self.optimizers: opt.zero_grad()
            total_loss.backward()
            for opt in self.optimizers: opt.step()

            predict = self.__model_predict(self.loss, logits)
            loss_list.append(total_loss)
            accuracy_list.append(self._get_accuracy(labels, predict))

        loss = sum(loss_list) / len(loss_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)

        if epoch > self.lr_scheduler_warm_up:
            self.lr_scheduler.step(loss)

        return loss.item(), accuracy

    def test(self) -> dict:
        self._load_model()
        self.model.eval()

        test_feature, test_label, test_length = self.test_data
        feats = test_feature.float().to(self.device)
        logits = test_label.long().to(self.device)
        out = self.model(feats).float()

        _, predict = torch.max(out, dim=1)
        test_result = {'test_accuracy': self._get_accuracy(logits, predict)}

        print('{0} - {1}'.format(self.__class__.__name__, test_result))
        return test_result

    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            for j in zip(i[0], i[1]):
                all += 1
                if j[0] == j[1]:
                    correct += 1
        return correct / all

    def __model_predict(self, kinds_loss, logits):
        if type(kinds_loss) == CRFLoss:
            return torch.tensor(self.loss.decode(logits))
        else:
            return torch.tensor(torch.max(logits, dim=1)[1])

    def __get_length(self, sequence):
        """
        pad는 [0...0]이니까 1더해서 [1...1]로
        만들고 all로 검사해서 pad가 아닌 부분만 세기
        """
        sequence = sequence.squeeze()
        return [all(map(int, (i + 1).tolist()))
                for i in sequence].count(False)
