import os

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from backend.decorators import intent
from backend.model.sklearn_models import SKLEARN_ALL_MODELS
from backend.proc.base.sklearn_processor import SklearnProcessor


@intent
class FallbackDetector(SklearnProcessor):

    def __init__(self):
        self.models = SKLEARN_ALL_MODELS
        super().__init__(self.models[0])

    def train(self, dataset):
        pipeline = Pipeline([('clf', self.models[0])])
        parameters = {'clf': self.models}

        distance, label = dataset
        gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(distance, label)

        for param_name in sorted(parameters.keys()):
            self.model = gs_clf.best_params_[param_name]
            print("best classifier : {0}, score : {1}"
                  .format(gs_clf.best_params_[param_name], gs_clf.best_score_))

        self.model.fit(distance, label)
        self._save_model()

    def test(self, dataset):
        self._load_model()

        distance, label = dataset
        predict = self.model.predict(distance)
        return self._get_accuracy(predict, label)

    def inference(self, dataset):
        self._load_model()

        distance = dataset
        predict = self.model.predict(distance)
        return predict

    def _load_model(self):
        if not self.model_loaded:
            self.model = joblib.load(self.model_dir + 'FallbackDetector.pkl')
            self.model_loaded = True

    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        joblib.dump(self.model, self.model_dir + 'FallbackDetector.pkl')
