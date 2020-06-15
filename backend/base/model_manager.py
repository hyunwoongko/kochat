
from backend import config
from backend.base.base_manager import BackendManager


class ModelManager(BackendManager):
    def __init__(self):
        super().__init__()
        for key, val in config.MODEL.items():
            setattr(self, key, val)


class Embedding(ModelManager):
    def __init__(self):
        super().__init__()
        for key, val in config.EMBEDDING.items():
            setattr(self, key, val)


class Intent(ModelManager):
    def __init__(self):
        super().__init__()
        for key, val in config.INTENT.items():
            setattr(self, key, val)


class Entity(ModelManager):
    def __init__(self):
        super().__init__()
        for key, val in config.ENTITY.items():
            setattr(self, key, val)
