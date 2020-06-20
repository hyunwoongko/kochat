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
        self.model = model
        super().__init__(self.model)

    def _load_model(self):
        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.model = joblib.load(self.model_file + '.pkl')
            self.model_loaded = True

    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        joblib.dump(self.model, self.model_file + '.pkl')

    def _print_log(self, msg):
        print('{name} - {msg}'
              .format(name=__class__.__name__, msg=msg))
