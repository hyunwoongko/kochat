"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import torch
import matplotlib.pyplot as plt
from collections import Counter
from torch.optim import Adam, SGD
from backend.decorators import intent
from backend.proc.torch_processor import TorchProcessor
from util.oop import override


@intent
class IntentRetrieval(TorchProcessor):
    def inference(self, sequence):
        pass

    def __init__(self, model, loss):
        super().__init__(model)
        self.loss = loss.to(self.device)
        self.label_dict = model.label_dict
        self.loss_optimizer = SGD(params=loss.parameters(), lr=self.loss_lr)
        self.model_optimizer = Adam(params=self.model.parameters(), lr=self.model_lr, weight_decay=self.weight_decay)
        self.optimizers = [self.loss_optimizer, self.model_optimizer]
        self.memory = None

    @override(TorchProcessor)
    def _train_epoch(self) -> tuple:
        losses, accuracies = [], []
        for train_feature, train_label in self.train_data:
            x = train_feature.float().to(self.device)
            y = train_label.long().to(self.device)
            feats = self.model(x)
            feats = self.model.ret_features(feats).float()
            logits = self.model.ret_logits(feats)

            loss = self.loss.step(logits, feats, y, self.optimizers)
            losses.append(loss.item())
            _, predict = torch.max(logits, dim=1)
            accuracies.append(self._get_accuracy(y, predict))

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        return loss, accuracy

    @override(TorchProcessor)
    def _test_epoch(self):
        # self.model.load_state_dict(torch.load(self.intent_retrieval_file))
        # self.model.eval()
        #
        # return {'classification_result': self._test_classification(),
        #         'retrieval_result': self._test_in_distribution()}
        return "NONE"

    @override(TorchProcessor)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all

    def _test_classification(self) -> float:
        self._load_model()
        test_feature, test_label = self.test_data
        x = test_feature.float().to(self.device)
        y = test_label.long().to(self.device)
        feats = self.model(x).to(self.device)
        feats = self.model.ret_features(feats)
        logits = self.model.clf_logits(feats)

        _, predict = torch.max(logits, dim=1)
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

        return self._get_accuracy(predict, label)

    def _draw_feature_space(self, feat, labels, epoch):
        feat = feat.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        data = np.c_[feat, labels]

        if self.d_loss == 2:
            data = pd.DataFrame(data, columns=['x_axis', 'y_axis', 'label'])
            ax = plt.figure().add_subplot()
            ax.scatter(data['x_axis'], data['y_axis'], marker='o', c=data['label'])
            plt.savefig(self.logs_dir + '{0}_{1}_2D.png'.format(self.model.name, epoch))
            plt.close()

        elif self.d_loss == 3:
            data = pd.DataFrame(data, columns=['x_axis', 'y_axis', 'z_axis', 'label'])
            ax = plt.figure().gca(projection='3d')
            ax.scatter(data['x_axis'], data['y_axis'], data['z_axis'], marker='o', c=data['label'])
            plt.savefig(self.logs_dir + '{0}_{1}_3D.png'.format(self.model.name, epoch))
            plt.close()

        else:
            print("2차원과 3차원만 피쳐스페이스 시각화가 가능합니다.")
            print("다른 차원의 경우 시각화는 무시하고 계산만 진행합니다.")
