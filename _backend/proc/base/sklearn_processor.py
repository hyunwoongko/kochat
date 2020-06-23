"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from abc import ABCMeta

import joblib

from _backend.proc.base.base_processor import BaseProcessor


class SklearnProcessor(BaseProcessor, metaclass=ABCMeta):
    def __init__(self, model):
        super().__init__(model)

    def _load_model(self):
        """
        저장된 모델을 불러옵니다.
        """

        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.model = joblib.load(self.model_file + '.pkl')
            self.model_loaded = True

    def _save_model(self):
        """
        모델을 저장장치에 저장합니다.
        """

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        joblib.dump(self.model, self.model_file + '.pkl')