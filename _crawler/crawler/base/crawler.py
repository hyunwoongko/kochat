import itertools
import re

from _crawler.decorators import crawler


@crawler
class Crawler:

    def untag(self, text):
        return re.sub(re.compile('<.*?>'), '', str(text))

    def flatten(self, list_to_flatten):
        return list(itertools.chain.from_iterable(list_to_flatten))
