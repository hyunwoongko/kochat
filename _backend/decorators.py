<<<<<<< HEAD
from _backend import config


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


def loss(cls):
    cls = backend(cls)
    for key, val in config.LOSS.items():
        setattr(cls, key, val)
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
=======
from _backend import config


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


def loss(cls):
    cls = backend(cls)
    for key, val in config.LOSS.items():
        setattr(cls, key, val)
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
>>>>>>> 998bcd017cd44db5c996455ee9ee1193cb11520e
