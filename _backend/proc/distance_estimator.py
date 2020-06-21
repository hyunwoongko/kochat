from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from _backend.decorators import intent
from _backend.proc.base.sklearn_processor import SklearnProcessor
import numpy as np


@intent
class DistanceEstimator(SklearnProcessor):
    def __init__(self, label_dict):
        super().__init__(None)
        self.label_dict = label_dict

    def predict(self, dataset):
        self._load_model()

        dataset = dataset.detach().cpu().numpy()
        dataset = np.expand_dims(dataset, 0)
        distance, _ = self.model.kneighbors(dataset)
        predict = self.model.predict(dataset)
        return predict, distance

    def fit(self, dataset):
        feats, label = dataset
        feats = feats.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        grid_search = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=self.dist_param,
            scoring='accuracy',
            n_jobs=-1)  # 모든 프로세서 사용
        grid_search.fit(feats, label)

        self.model = KNeighborsClassifier(
            n_neighbors=grid_search.best_params_['n_neighbors'],
            weights=grid_search.best_params_['weights'],
            p=grid_search.best_params_['p'],
            algorithm=grid_search.best_params_['algorithm'],
            n_jobs=-1)  # 모든 프로세서 사용
        self.model.fit(feats, label)
        self._save_model()

    def make_dist_dataset(self, dataset):
        self._load_model()

        feats, label = dataset
        feats = feats.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        distance, _ = self.model.kneighbors(feats)

        label_set = []
        for i in label:
            if i < len(self.label_dict):
                label_set.append(0)  # in distribution
            else:
                label_set.append(1)  # out distribution (open / close)

        return distance, label_set

    def test(self, dataset):
        self._load_model()
        feats, label = dataset
        feats = feats.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        predict = self.model.predict(feats)
        return self._get_accuracy(label, predict)
