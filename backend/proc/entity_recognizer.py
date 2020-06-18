"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from backend.decorators import entity
from backend.loss.softmax_loss import SoftmaxLoss
from backend.proc.base.torch_processor import TorchProcessor
from util.oop import override


@entity
class EntityRecognizer(TorchProcessor):

    def __init__(self, model):
        super().__init__(model)
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

    @override(TorchProcessor)
    def _train(self, epoch) -> tuple:
        losses, accuracies = [], []
        for train_feature, train_label in self.train_data:
            x = train_feature.float().to(self.device)
            y = train_label.long().to(self.device)
            out = self.model(x).float()

            total_loss = self.loss.compute_loss(out, None, y)
            self.loss.step(total_loss, self.optimizers)

            losses.append(total_loss.item())
            _, predict = torch.max(out, dim=1)
            accuracies.append(self._get_accuracy(y, predict))

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)

        if epoch > self.lr_scheduler_warm_up:
            self.lr_scheduler.step(loss)

        return loss, accuracy

    @override(TorchProcessor)
    def test(self) -> dict:
        self._load_model()
        self.model.eval()

        test_feature, test_label = self.test_data
        x = test_feature.float().to(self.device)
        y = test_label.long().to(self.device)
        out = self.model(x).float()

        _, predict = torch.max(out, dim=1)
        test_result = {'test_accuracy': self._get_accuracy(y, predict)}
        print(test_result)
        return test_result

    @override(TorchProcessor)
    def inference(self, sequence):
        self._load_model()
        self.model.eval()

        length = self._get_length(sequence)
        output = self.model(sequence).float()
        output = output.squeeze().t()
        _, predict = torch.max(output, dim=1)
        output = [list(self.label_dict.keys())[i.item()] for i in predict]
        return ' '.join(output[:length])

    def _get_length(self, sequence):
        """
        pad는 [0...0]이니까 1더해서 [1...1]로
        만들고 all로 검사해서 pad가 아닌 부분만 세기
        """
        sequence = sequence.squeeze()
        return [all(map(int, (i + 1).tolist()))
                for i in sequence].count(False)

    @override(TorchProcessor)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            for j in zip(i[0], i[1]):
                all += 1
                if j[0] == j[1]:
                    correct += 1
        return correct / all
