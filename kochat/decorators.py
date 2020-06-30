import kochat_config


def backend(cls):
    for key, val in kochat_config.BASE.items():
        setattr(cls, key, val)
    return cls


def data(cls):
    cls = backend(cls)
    for key, val in kochat_config.DATA.items():
        setattr(cls, key, val)
    return cls


def proc(cls):
    cls = backend(cls)
    for key, val in kochat_config.PROC.items():
        setattr(cls, key, val)

    return cls


def loss(cls):
    cls = backend(cls)
    for key, val in kochat_config.LOSS.items():
        setattr(cls, key, val)
    return cls


def gensim(cls):
    cls = backend(cls)
    for key, val in kochat_config.GENSIM.items():
        setattr(cls, key, val)
    return cls


def intent(cls):
    cls = backend(cls)
    for key, val in kochat_config.INTENT.items():
        setattr(cls, key, val)
    return cls


def entity(cls):
    cls = backend(cls)
    for key, val in kochat_config.ENTITY.items():
        setattr(cls, key, val)
    return cls
