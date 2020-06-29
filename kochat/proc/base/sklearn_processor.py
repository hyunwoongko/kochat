"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from typing import Any

import joblib
from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings
from kochat.proc.base.base_processor import BaseProcessor


class SklearnProcessor(BaseProcessor):

    def __init__(self, model: BaseEstimator):
        """
        Sklearn 모델의 Training, Testing, Inference
        등을 관장하는 프로세서 클래스입니다.

        Sklearn 모델은 Intent, Entity 등의 주요기능을 구현하기보다는 주로
        Fallback Detection, Distance Estimation 등의 서브기능을 구현하기 위해 사용합니다.

        :param model: Sklearn 모델을 입력해야합니다.
        """

        super().__init__(model)

    @abstractmethod
    @ignore_warnings(category=Warning)
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    @ignore_warnings(category=Warning)
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @ignore_warnings(category=Warning)
    def _load_model(self):
        """
        저장된 모델을 불러옵니다.
        """

        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.model = joblib.load(self.model_file + '.pkl')
            self.model_loaded = True

    @ignore_warnings(category=Warning)
    def _save_model(self):
        """
        모델을 저장장치에 저장합니다.
        """

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        joblib.dump(self.model, self.model_file + '.pkl')

    @ignore_warnings(category=Warning)
    def _grid_search(self, feats: Any, label: Any):
        pass  # 반드시 구현할 필요는 없음.
