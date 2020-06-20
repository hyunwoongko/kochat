import urllib
from abc import ABCMeta, abstractmethod
from urllib.request import urlopen, Request

import bs4

from _crawler.crawler.base.crawler import Crawler
from _crawler.decorators import searcher


@searcher
class Searcher(Crawler, metaclass=ABCMeta):

    @abstractmethod
    def _make_query(self, *args, **kwargs):
        raise NotImplementedError

    def _naver(self, query, selector):
        url = self.url['naver'] + urllib.parse.quote(query)
        out = bs4.BeautifulSoup(urlopen(Request(url)).read(), 'html.parser')
        try:
            return [s.contents for s in out.select(selector)]
        except:
            return None
