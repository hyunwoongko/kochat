from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from backend.decorators import intent
from backend.proc.base.sklearn_processor import SklearnProcessor
import numpy as np


@intent
class DistanceEstimator(SklearnProcessor):
    def __init__(self, model, label_dict):
        super().__init__(model)
        self.label_dict = label_dict

    def train(self, dataset):
        x, y = dataset['X'], dataset['Y']

        grid_search = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=self.dist_param,
            cv=10, n_jobs=-1,  # 모든 프로세서 사용
            scoring='accuracy')
        grid_search.fit(x, y)
        self.dist_param = grid_search.best_params_
        print(self.dist_param)

        self.model = KNeighborsClassifier(
            n_neighbors=self.dist_param['n_neighbors'],
            weights=self.dist_param['weights'],
            p=self.dist_param['p'], n_jobs=-1,  # 모든 프로세서 사용
            algorithm=self.dist_param['algorithm'])
        self.model.fit(x, y)
        self._save_model()

    def test(self, dataset):
        self._load_model()

        feats, label = dataset
        feats = feats.detach().cpu()
        label = label.detach().cpu().numpy()
        predict = self.model.predict(feats)
        return self._get_accuracy(label, predict)

    def inference(self, dataset):
        self._load_model()
        dataset = np.expand_dims(dataset, 0)

        distance, _ = self.model.kneighbors(dataset)
        predict = self.model.predict(dataset)
        return predict, distance

    def make_dist_dataset(self, dataset):
        self._load_model()

        feats, label = dataset
        distance, _ = self.model.kneighbors(feats)
        label_set = []
        for i in label:
            if i < len(self.label_dict):
                label_set.append(0)
                # in distribution
            else:
                label_set.append(1)
                # out distribution (open / close)

        return distance, label_set
