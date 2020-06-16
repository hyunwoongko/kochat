from abc import abstractmethod, ABCMeta
from backend import config
from backend.decorators import backend


@backend
class BaseProcessor(metaclass=ABCMeta):
    def __init__(self, model):
        super().__init__()
        for key, val in config.PROC.items():
            setattr(self, key, val)

        self.train_data = None
        self.test_data = None

        self.model = model
        self.model_file: str = model.save_path()
        self.model_dir: list = self.model_file.split('/')
        del self.model_dir[len(self.model_dir) - 1]
        self.model_dir: str = '/'.join(self.model_dir)
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
