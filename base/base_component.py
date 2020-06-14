import config


class BaseComponent:
    def __init__(self):
        for key, val in config.BASE.items():
            setattr(self, key, val)


def override(super_class):
    def overrider(method):
        assert (method.__name__ in dir(super_class))
        return method

    return overrider
