import re

from crawler import config
from crawler.base.base import Crawler


class Editor(Crawler):
    def __init__(self):
        super().__init__()
        for key, val in config.EDIT.items():
            setattr(self, key, val)

    def edit(self, list_, dict_):
        """
        only support 1D
        """
        return [re.sub(word, dict_[word], word) for word in list_]
