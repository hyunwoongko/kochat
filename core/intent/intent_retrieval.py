from collections import Counter

import torch

from base.model_managers.model_manager import Intent


class IntentRetrieval(Intent):

    def __init__(self, model, label_dict):
        super().__init__()
        self.label_dict = label_dict
        self.model = model.Model(vector_size=self.vector_size,
                                 max_len=self.max_len,
                                 d_model=self.d_model,
                                 layers=self.layers,
                                 classes=len(self.label_dict))

        self.train_neighbors = []
        self.model.load_state_dict(torch.load(self.intent_retrieval_file))
        self.model.eval()

        for train_feature, train_label in self.train_data:
            for sample in zip(train_feature, train_label):
                x, y = sample[0].cuda(), sample[1].cuda()
                y_ = self.model(x.unsqueeze(0))
                self.train_neighbors.append((y_, y))

    def _test_in_distribution(self) -> float:
        predict, label, train_neighbors = [], [], []
        self.model.load_state_dict(torch.load(self.intent_retrieval_file))
        self.model.eval()

        for train_feature, train_label in self.train_data:
            for sample in zip(train_feature, train_label):
                x, y = sample[0].cuda(), sample[1].cuda()
                y_ = self.model(x.unsqueeze(0))
                train_neighbors.append((y_, y))

        test_feature, test_label = self.test_data
        for i, sample in enumerate(zip(test_feature, test_label)):
            x, y = sample[0].cuda(), sample[1].cuda()
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

        return self._get_accuracy(predict, label)
