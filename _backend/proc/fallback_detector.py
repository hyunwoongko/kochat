from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings

from _backend.decorators import intent
from _backend.proc.base.sklearn_processor import SklearnProcessor


@intent
class FallbackDetector(SklearnProcessor):

    def __init__(self):
        self.models = self.fallback_detector
        super().__init__(self.models[0])

    def predict(self, dataset):
        self._load_model()
        distance = dataset
        predict = self.model.predict(distance)
        return predict

    @ignore_warnings(category=Warning)
    def fit(self, dataset):
        pipeline = Pipeline([('clf', self.models[0])])
        parameters = {'clf': self.models}

        distance, label = dataset
        gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(distance, label)

        for param_name in sorted(parameters.keys()):
            self.model = gs_clf.best_params_[param_name]
            self._print_log("best classifier: {0}, score : {1}"
                            .format(gs_clf.best_params_[param_name], gs_clf.best_score_))

        self.model.fit(distance, label)
        self._save_model()

    def test(self, dataset):
        self._load_model()
        distance, label = dataset
        predict = self.model.predict(distance)
        return self._get_accuracy(predict, label)
