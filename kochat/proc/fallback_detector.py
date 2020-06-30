import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings

from kochat.decorators import intent
from kochat.proc.sklearn_processor import SklearnProcessor


@intent
class FallbackDetector(SklearnProcessor):

    def __init__(self, label_dict: dict, grid_search: bool = True):
        """
        OOD 데이터셋이 존재하는 경우,
        In distribution 데이터와 Out dist
        :param grid_search:
        """

        self.model = self.fallback_detectors[0]
        self.label_dict = label_dict
        self.grid_search = grid_search
        super().__init__(self.model)

    @ignore_warnings(category=Warning)
    def fit(self, feats, label, mode: str):
        """
        Fallback Detector를 학습 및 검증합니다.

        :param feats: 학습할 피쳐
        :param label: 라벨 리스트
        :param mode: train or test
        :return: mode가 test인 경우, predicts와 label을 리턴합니다.
        """
        if not isinstance(label, np.ndarray):
            label = label.detach().cpu().numpy()
        if not isinstance(feats, np.ndarray):
            feats = feats.detach().cpu().numpy()

        binary_label_set = []

        for i in label:
            if i >= 0:
                # in distribution (0 이상)
                binary_label_set.append(0)
            else:
                # out distribution (-1)
                binary_label_set.append(1)

        binary_label_set = np.array(binary_label_set)
        binary_label_set = np.expand_dims(binary_label_set, axis=1)

        if mode == 'train':
            self._train_epoch(feats, binary_label_set)

        else:
            predicts = self._test_epoch(feats)
            return predicts, binary_label_set

    @ignore_warnings(category=Warning)
    def predict(self, feats) -> np.ndarray:
        """
        사용자의 입력에 inference합니다.

        :param feats: 학습할 피쳐
        :return: Fallback 여부 반환
        """

        if not isinstance(feats, np.ndarray):
            feats = feats.detach().cpu().numpy()

        self._load_model()

        predicts = self._test_epoch(feats)
        return predicts

    @ignore_warnings(category=Warning)
    def _train_epoch(self, feats: np.ndarray, label: np.ndarray):
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.
        grid_search가 True인 경우 grid search를 수행합니다.

        :param feats: 학습할 피쳐
        :param label: 라벨 리스트
        """

        if self.grid_search:
            self.model = self._grid_search(feats, label.ravel())
        else:
            self.model.fit(feats, label.ravel())

        self._save_model()

    @ignore_warnings(category=Warning)
    def _test_epoch(self, feats: np.ndarray) -> np.ndarray:
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.

        :param feats: 학습할 피쳐
        :return: Fallback 여부 반환
        """

        predicts = self.model.predict(feats)
        return predicts

    @ignore_warnings(category=Warning)
    def _grid_search(self, feats: np.ndarray, label: np.ndarray) -> BaseEstimator:
        pipeline = Pipeline([('detector', self.model)])
        parameters = {'detector': self.fallback_detectors}

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(feats, label)
        detector = grid_search.best_params_['detector']

        detector.fit(feats, label)
        return detector
