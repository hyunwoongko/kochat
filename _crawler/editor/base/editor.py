import re

from _crawler.crawler.base.crawler import Crawler
from _crawler.decorators import editor


@editor
class Editor(Crawler):

    def join_dict(self, _dict, key, insert=""):
        if _dict[key] is not None and len(_dict[key]) != 0:
            if _dict[key][0] is not None:
                _dict[key] = ' '.join(_dict[key])
                _dict[key] += insert
        return _dict

    def enumerate_josa(self, j1, j2, list_):
        josa = [j1]
        for i in range(len(list_) - 1):
            if list_[i] == list_[i + 1]:
                josa.append(j2)
            else:
                josa.append(j1)

        return josa
