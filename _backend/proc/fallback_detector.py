from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from _backend.decorators import intent
from _backend.proc.base.sklearn_processor import SklearnProcessor


@intent
class FallbackDetector(SklearnProcessor):

    def __init__(self, label_dict, grid_search=True):
        """

        :param grid_search:
        """

        self.model = self.fallback_detectors[0]
        self.label_dict = label_dict
        self.grid_search = grid_search
        super().__init__(self.model)

    def fit(self, distance, label):
        """
        Fallback Detector를 학습 및 검증합니다.

        :param distance: Distance Estimator의 출력
        :param label: 라벨 리스트
        """

        binary_label_set = []

        for i in label:
            if i < len(self.label_dict):
                # in distribution
                binary_label_set.append(0)
            else:
                # out distribution
                binary_label_set.append(1)

        self._train_epoch(distance, binary_label_set)

    def predict(self, distance):
        """
        사용자의 입력에 inference합니다.

        :param distance: Distance Estimator의 출력
        :return: Fallback 여부 반환
        """

        predicts = self._test_epoch(distance)
        return predicts

    def _train_epoch(self, distance, label):
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.
        grid_search가 True인 경우 grid search를 수행합니다.

        :param distance: Distance Estimator의 출력
        :param label: 라벨 리스트
        """

        if self.grid_search:
            self.model = self._grid_search(distance, label)
        else:
            self.model.fit(distance, label)

        self._save_model()

    def _test_epoch(self, distance):
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.

        :param distance: Distance Estimator의 출력
        :return: Fallback 여부 반환
        """

        predicts = self.model.predict(distance)
        return predicts

    def _grid_search(self, distance, label):
        pipeline = Pipeline([('detector', self.model)])
        parameters = {'detector': self.fallback_detectors}

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(distance, label)
        return grid_search.best_estimator_
