from abc import abstractmethod, ABCMeta

from _backend.decorators import proc


@proc
class BaseProcessor(metaclass=ABCMeta):
    def __init__(self, model):
        super().__init__()
        self.train_data = None
        self.test_data = None

        self.model = model
        self.model_loaded = False
        self.model_dir = self.model_dir + \
                         self.__class__.__name__ + \
                         self.delimeter

        self.model_file: str = self.model_dir + \
                               self.delimeter + \
                               self.__class__.__name__

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
