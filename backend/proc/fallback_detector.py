from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings

from backend.decorators import intent
from backend.model.sklearn_models import LINEAR_MODELS
from backend.proc.base.sklearn_processor import SklearnProcessor


@intent
class FallbackDetector(SklearnProcessor):

    def __init__(self):
        self.models = LINEAR_MODELS
        super().__init__(self.models[0])
        self.make_anonymous()

    def predict(self, dataset):
        self._load_model()

        distance = dataset
        predict = self.model.predict(distance)
        return predict

    @ignore_warnings(category=Warning)
    def fit(self, dataset):
        self._print_log("msg: start train ...")
        pipeline = Pipeline([('clf', self.models[0])])
        parameters = {'clf': self.models}

        distance, label = dataset
        gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(distance, label)

        for param_name in sorted(parameters.keys()):
            self.model = gs_clf.best_params_[param_name]
            self.make_anonymous()

            self._print_log("best classifier: {0}, score : {1}"
                            .format(gs_clf.best_params_[param_name], gs_clf.best_score_))

        self.model.fit(distance, label)
        self._save_model()

    def test(self, dataset):
        self._print_log("msg: start test ...")
        self._load_model()

        distance, label = dataset
        predict = self.model.predict(distance)
        return self._get_accuracy(predict, label)

    def make_anonymous(self):
        self.model.name = 'FallbackDetector'
        self.set_save_path(self.model, name=self.model.name)
