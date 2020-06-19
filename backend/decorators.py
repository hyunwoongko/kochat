from backend import config


def backend(cls):
    for key, val in config.BACKEND.items():
        setattr(cls, key, val)
    return cls


def data(cls):
    cls = backend(cls)
    for key, val in config.DATA.items():
        setattr(cls, key, val)

    return cls


def proc(cls):
    cls = backend(cls)
    for key, val in config.PROC.items():
        setattr(cls, key, val)

    return cls


def model(cls):
    cls = backend(cls)
    for key, val in config.MODEL.items():
        setattr(cls, key, val)

    def save_dir(self):
        module = cls.__module__.split('.')
        module = module[len(module) - 1]
        return self.model_dir + module + '/'

    def save_file(self, name=None):
        if name is None:
            return save_dir(self) + cls.__name__
        else:
            return save_dir(self) + name

    setattr(cls, 'save_dir', save_dir)
    setattr(cls, 'save_file', save_file)
    setattr(cls, 'name', cls.__name__)
    return cls


def loss(cls):
    cls = backend(cls)
    for key, val in config.LOSS.items():
        setattr(cls, key, val)

    setattr(cls, 'name', cls.__name__)
    return cls


def gensim(cls):
    cls = backend(cls)
    for key, val in config.GENSIM.items():
        setattr(cls, key, val)
    return cls


def intent(cls):
    cls = backend(cls)
    for key, val in config.INTENT.items():
        setattr(cls, key, val)
    return cls


def entity(cls):
    cls = backend(cls)
    for key, val in config.ENTITY.items():
        setattr(cls, key, val)
    return cls
