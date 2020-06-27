import json
import urllib
from abc import ABCMeta, abstractmethod
from urllib.request import urlopen, Request

import bs4
import requests

from _crawler.crawler.base.base_crawler import BaseCrawler
from _crawler.decorators import searcher


@searcher
class BaseSearcher(BaseCrawler, metaclass=ABCMeta):

    @abstractmethod
    def _make_query(self, *args, **kwargs):
        raise NotImplementedError

    def __bs4(self, url: str, query: str) -> bs4.BeautifulSoup:
        """
        beautiful soup 4를 이용하여 정적 웹페이지에 대한 크롤링을 시도합니다.

        :param url: 베이스 url
        :param query: 검색할 쿼리
        :return: parsing된 html
        """

        url = url + urllib.parse.quote(query)
        out = bs4.BeautifulSoup(urlopen(Request(url, headers=self.headers)).read(), 'html.parser')
        return out

    def _bs4_contents(self, url: str, selectors: list, query: str = ""):
        """
        beautiful soup 4를 이용하여 정적 웹페이지에 대한 크롤링을 시도합니다.
        셀렉터를 적용하여 입력한 셀렉터에 해당하는 태그 안의 contents를 로드합니다.

        :param url: 베이스 url
        :param selectors: 검색할 셀렉터
        :param query: 검색할 쿼리
        :return: 크롤링된 콘텐츠
        """

        out = self.__bs4(url, query)
        try:
            crawled = []
            for selector in selectors:
                for s in out.select(selector):
                    crawled.append(s.contents)
            return crawled
        except Exception:
            return None

    def _bs4_documents(self, url: str, selectors: list, query: str = ""):
        """
        beautiful soup 4를 이용하여 정적 웹페이지에 대한 크롤링을 시도합니다.
        셀렉터를 적용하여 입력한 셀렉터에 해당하는 태그를 포함한 모든 document 구조를 로드합니다.

        :param url: 베이스 url
        :param selectors: 검색할 셀렉터
        :param query: 검색할 쿼리
        :return: 크롤링된 콘텐츠
        """

        out = self.__bs4(url, query)
        try:
            crawled = []
            for selector in selectors:
                for s in out.select(selector):
                    crawled.append(s)
            return crawled
        except Exception:
            return None

    def _json(self, url: str, query: str):
        """
        json을 이용하여 동적 웹페이지에 대한 크롤링을 시도합니다.

        :param url: 베이스 url
        :param query: 검색할 쿼리
        :return: 크롤링된 json 파일
        """

        url += urllib.parse.quote(query)
        req = requests.get(url, headers=self.headers)
        if req.status_code == requests.codes.ok:
            loaded_data = json.loads(req.text)
            return loaded_data
        else:
            return None
