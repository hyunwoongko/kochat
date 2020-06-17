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


def model(cls):
    cls = backend(cls)
    for key, val in config.MODEL.items():
        setattr(cls, key, val)

    def save_path(self):
        name = cls.__module__.split('.')
        name = name[len(name) - 1]
        return self.model_dir + name + '.pth'

    setattr(cls, 'name', cls.__name__)
    setattr(cls, 'save_path', save_path)
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
