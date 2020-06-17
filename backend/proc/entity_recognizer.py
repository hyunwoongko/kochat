"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from backend.decorators import entity
from backend.proc.torch_processor import TorchProcessor
from util.oop import override


@entity
class EntityRecognizer(TorchProcessor):

    def __init__(self, model, label_dict):
        super().__init__(model)
        self.label_dict = label_dict
        self.loss = CrossEntropyLoss()
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.model_lr,
            weight_decay=self.weight_decay)

    @override(TorchProcessor)
    def _train_epoch(self, train_data, test_data) -> tuple:
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

    @override(TorchProcessor)
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

    @override(TorchProcessor)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            for j in zip(i[0], i[1]):
                all += 1
                if j[0] == j[1]:
                    correct += 1
        return correct / all

    def inference_model(self, sequence):
        length = self.get_length(sequence)
        output = self.model(sequence).float()
        output = output.squeeze().t()
        _, predict = torch.max(output, dim=1)
        output = [list(self.label_dict.keys())[i.item()] for i in predict]
        return ' '.join(output[:length])

    def get_length(self, sequence):
        """
        pad는 [0...0]이니까 1더해서 [1...1]로
        만들고 all로 검사해서 pad가 아닌 부분만 세기
        """
        sequence = sequence.squeeze()
        return [all(map(int, (i + 1).tolist()))
                for i in sequence].count(False)
