import config


class BaseComponent:
    def __init__(self):
        for key, val in config.BASE.items():
            setattr(self, key, val)
