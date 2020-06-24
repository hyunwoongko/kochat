import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from _backend.decorators import intent
from _backend.proc.base.sklearn_processor import SklearnProcessor


@intent
class DistanceEstimator(SklearnProcessor):

    def __init__(self, grid_search=True):
        """
        Nearest Neighbors 알고리즘을 기반으로 가장 가까운 K개의 샘플을 검색한뒤
        가장 많이 검색된 클래스로 분류하고, 샘플들과의 거리를 출력하는 클래스입니다.

        :param grid_search: 그리드 서치 사용 여부
        """

        self.model = KNeighborsClassifier(n_neighbors=10)
        self.grid_search = grid_search
        super().__init__(self.model)

    def fit(self, feats, label, mode):
        """
        Distance Estimator를 학습 및 검증합니다.

        :param feats: features
        :param label: 라벨 리스트
        :param mode: train or test
        :return: predict, K개 sample에 대한 distances
        """

        feats = feats.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        if mode == 'train':
            self._train_epoch(feats, label)

        predict, distance = self._test_epoch(feats)
        return predict, distance

    def predict(self, sequence):
        """
        사용자의 입력에 inference합니다.

        :param sequence: 입력 시퀀스
        :return: 분류결과와 가장 가까운 K개의 샘플과의 거리
        """

        sequence = sequence.detach().cpu().numpy()
        sequence = np.expand_dims(sequence, axis=0)

        predict, distance = self._test_epoch(sequence)
        return predict, distance

    def _train_epoch(self, feats, label):
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.
        grid_search가 True인 경우 grid search를 수행합니다.

        :param feats: 입력 features
        :param label: 라벨 리스트
        """

        if self.grid_search:
            self.model = self._grid_search(feats, label)
        else:
            self.model.fit(feats, label)

        self._save_model()

    def _test_epoch(self, feats):
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.

        :param epoch: 입력 features
        :return: 분류결과와 가장 가까운 K개의 샘플과의 거리
        """

        self._load_model()

        predict = self.model.predict(feats)
        distance, _ = self.model.kneighbors(feats)
        return predict, distance

    def _grid_search(self, feats, label):
        """
        가장 적합한 K와 여러가지 파라미터를 선택하기 위해 그리드 서치를 진행합니다.

        :param feats: 다른 모델 등으로부터 출력된 features
        :param label: 라벨 리스트
        :return: search된 best estimator
        """

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.dist_param,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(feats, label)
        return grid_search.best_estimator_
