from crawler import config
from crawler.base.base import Crawler


class Answerer(Crawler):

    def __init__(self):
        super().__init__()
        for key, val in config.ANSWER.items():
            setattr(self, key, val)

    def sorry(self, text=None):
        if text is None:
            return self.fallback
        else:
            return text
