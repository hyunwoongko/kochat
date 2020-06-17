from abc import abstractmethod, ABCMeta

from backend.decorators import backend


@backend
class BaseProcessor(metaclass=ABCMeta):
    def __init__(self, model):
        super().__init__()
        self.train_data = None
        self.test_data = None

        self.model = model
        self.model_dir: str = model.save_dir()
        self.model_file: str = model.save_file()
        self.model_loaded = False

    @abstractmethod
    def train(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def inference(self, sequence):
        raise NotImplementedError

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError

    @abstractmethod
    def _save_model(self):
        raise NotImplementedError
