import itertools
import re

from crawler import config


class Crawler:
    def __init__(self):
        for key, val in config.CRAWLER.items():
            setattr(self, key, val)

    def untag(self, text):
        return re.sub(re.compile('<.*?>'), '', str(text))

    def flatten(self, list_to_flatten):
        return list(itertools.chain.from_iterable(list_to_flatten))
