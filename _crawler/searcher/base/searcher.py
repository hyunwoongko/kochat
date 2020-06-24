import json
import urllib
from abc import ABCMeta, abstractmethod
from urllib.request import urlopen, Request

import bs4
import requests

from _crawler.crawler.base.crawler import Crawler
from _crawler.decorators import searcher


@searcher
class Searcher(Crawler, metaclass=ABCMeta):

    @abstractmethod
    def _make_query(self, *args, **kwargs):
        raise NotImplementedError

    def __BS4(self, base_url, query):
        url = base_url + urllib.parse.quote(query)
        out = bs4.BeautifulSoup(urlopen(Request(url, headers=self.headers)).read(), 'html.parser')
        return out

    def _bs4_contents(self, url, selectors, query=""):
        out = self.__BS4(url, query)
        try:
            crawled = []
            for selector in selectors:
                for s in out.select(selector):
                    crawled.append(s.contents)
            return crawled
        except:
            return None

    def _bs4_documents(self, url, selectors, query=""):
        out = self.__BS4(url, query)
        try:
            crawled = []
            for selector in selectors:
                for s in out.select(selector):
                    crawled.append(s)
            return crawled
        except:
            return None

    def _ajax_json(self, url, query):
        url += urllib.parse.quote(query)
        req = requests.get(url, headers=self.headers)
        if req.status_code == requests.codes.ok:
            loaded_data = json.loads(req.text)
            return loaded_data
        else:
            return None
