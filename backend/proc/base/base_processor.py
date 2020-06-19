from abc import abstractmethod, ABCMeta

from backend.decorators import proc


@proc
class BaseProcessor(metaclass=ABCMeta):
    def __init__(self, model):
        super().__init__()
        self.train_data = None
        self.test_data = None

        self.model = model
        self.set_save_path(self.model)
        self.model_loaded = False

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def test(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError

    @abstractmethod
    def _save_model(self):
        raise NotImplementedError

    def _get_accuracy(self, predict, label) -> float:
        all, correct = 0, 0
        for i in zip(predict, label):
            all += 1
            if i[0] == i[1]:
                correct += 1
        return correct / all

    def set_save_path(self, model, name=None):
        self.model_dir: str = model.save_dir()
        self.model_file: str = model.save_file(name)
