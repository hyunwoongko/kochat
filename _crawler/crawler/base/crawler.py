import itertools
import re

import numpy as np

from _crawler.decorators import crawler


@crawler
class Crawler:

    def _untag(self, text):
        return re.sub(re.compile('<.*?>'), '', str(text))

    def _flatten_list(self, list_):
        np.array(list_)
        while len(np.array(list_).shape) != 1:
            list_ = list(itertools.chain.from_iterable(list_))
        return list_

    def _flatten_dicts(self, dict_):
        for k, v in dict_.items():
            dict_[k] = self._flatten_list(dict_[k])
        return dict_
