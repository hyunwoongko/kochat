"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import Adam, SGD

from backend.decorators import intent
from backend.proc.torch_processor import TorchProcessor
from util.oop import override, overload


@intent
class IntentRetrieval(TorchProcessor):

    def __init__(self, model, loss):
        super().__init__(model)
        self.loss = loss.to(self.device)
        self.label_dict = model.label_dict
        self.loss_optimizer = SGD(params=loss.parameters(), lr=self.loss_lr)
        self.model_optimizer = Adam(
            params=self.model.parameters(),
            lr=self.model_lr,
            weight_decay=self.weight_decay)

        self.optimizers = [self.loss_optimizer, self.model_optimizer]
        self.nearest_neighbors = KNeighborsClassifier()
        self.memory = {}

    @overload(torch.Tensor, torch.Tensor)
    def train(self, train_dataset, ood_dataset):
        """
        TorchProcessor의 train 함수 오버로딩
        기존 함수를 호출하면 classification test와 in distribution test만 진행
        재정의한 이 함수를 호출하여 OOD 데이터셋을 함께 넣어주면 OOD 테스트도 진행함
        """
        super(IntentRetrieval, self).train(train_dataset)

    @override(TorchProcessor)
    def inference(self, sequence):
        self._load_model()
        logits = self.model(sequence).float()
        logits = self.model.clf_logits(logits.squeeze())
        _, predict = torch.max(logits, dim=0)
        return list(self.label_dict)[predict.item()]

    @override(TorchProcessor)
    def _train_epoch(self, epoch) -> tuple:
        losses, accuracies, memories, labels = [], [], [], []
        for train_feature, train_label in self.train_data:
            x = train_feature.float().to(self.device)
            y = train_label.long().to(self.device)
            feats = self.model(x)
            feats = self.model.ret_features(feats).float()
            logits = self.model.ret_logits(feats)

            total_loss = self.loss.compute_loss(logits, feats, y)
            self.loss.step(total_loss, self.optimizers)
            memories.append(feats)
            labels.append(y)

            losses.append(total_loss.item())
            _, predict = torch.max(logits, dim=1)
            accuracies.append(self._get_accuracy(y, predict))

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        self.memory['X'] = torch.cat(memories, dim=0).detach().cpu().numpy()
        self.memory['Y'] = torch.cat(labels, dim=0).detach().cpu().numpy()

        if epoch % self.visualization_epoch == 0:
            self._draw_feature_space(self.memory['X'], self.memory['Y'], epoch)

        return loss, accuracy

    @override(TorchProcessor)
    def _test_epoch(self):
        self._load_model()
        test_feature, test_label = self.test_data
        x = test_feature.float().to(self.device)
        y = test_label.long().to(self.device)
        feats = self.model(x).to(self.device)
        feats = self.model.ret_features(feats)
        logits = self.model.ret_logits(feats)

        _, predict = torch.max(logits, dim=1)
        clf_result = self._get_accuracy(y, predict)

        feats = feats.detach().cpu()
        label = y.detach().cpu().numpy()
        predict = self.nearest_neighbors.predict(feats)
        ret_result = self._get_accuracy(label, predict)

        return {'softmax_classifier': clf_result,
                'knn_in_distribution_retrieval': ret_result}

    @override(TorchProcessor)
    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        params = self.knn_param_grid
        grid_search = GridSearchCV(
            estimator=self.nearest_neighbors,
            param_grid=params,
            cv=10, n_jobs=-1,  # 모든 프로세서 사용
            scoring='accuracy')

        grid_search.fit(self.memory['X'], self.memory['Y'])
        params = grid_search.best_params_
        self.nearest_neighbors = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            p=params['p'], n_jobs=-1,  # 모든 프로세서 사용
            algorithm=params['algorithm'])

        self.nearest_neighbors = KNeighborsClassifier()
        self.nearest_neighbors.fit(self.memory['X'], self.memory['Y'])
        joblib.dump(self.nearest_neighbors, self.model_file + 'pkl')
        torch.save(self.model.state_dict(), self.model_file + '.pth')
        print("Best Param : {}".format(params))

    @override(TorchProcessor)
    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all

    def _draw_feature_space(self, feat, labels, epoch):
        if self.d_loss == 2:  # 2차원인 경우
            data = np.c_[feat, labels]
            data = pd.DataFrame(data, columns=['x', 'y', 'label'])
            ax = plt.figure().add_subplot()
            ax.scatter(data['x'], data['y'], marker='o', c=data['label'])

        else:
            if self.d_loss > 3:
                # 4차원 이상인 경우 PCA로 3차원으로 만듬
                pca = PCA(n_components=3)
                feat = pca.fit_transform(feat)

            data = np.c_[feat, labels]
            data = pd.DataFrame(data=data, columns=['x', 'y', 'z', 'label'])
            ax = plt.figure().gca(projection='3d')
            ax.scatter(data['x'], data['y'], data['z'], marker='o', c=data['label'])

        file_name = self.model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        file_name += '{model}_{loss}_{dim}D_{epoch}.png' \
            .format(model=self.model.name,
                    loss=self.loss.name,
                    dim=self.d_loss,
                    epoch=epoch)
        plt.savefig(file_name)
        plt.close()
