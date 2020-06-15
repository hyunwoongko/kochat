import urllib
from urllib.request import urlopen, Request

import bs4
import numpy as np

from web_crawler.base.base_manager import SearchManager
from web_crawler.core.editor import Editor


class Searcher(SearchManager):
    """
    유저의 요청사항을 웹에 검색해서 정보를 얻음
    이후 모아온 정보들을 수정하기 위해 Editor에게 위임
    """

    def __init__(self):
        super().__init__()
        self.editor = Editor()

    def dust(self, date, location, old=False):
        query = [' '.join([date, location] + [i]) for i in self.intent['dust']]

        if old:
            if date in self.date['today']:
                today_selector = '.all_state > .state_list > li > .state_info'
                today_selector = [self._naver(q, today_selector) for q in query]

        else:
            if date in self.date['today'] + self.date['tomorrow'] + self.date['after']:
                selector = '.on.now > em'
                results = [self._naver(q, selector) for q in query]

                if len(sum(results, [])) != 18:
                    # 신버전의 경우 .on.now가 총 18개 나오고
                    # 구버전의 경우 .on.now가 2~3개 내외로 나옴.
                    return self.dust(date, location, old=True)

                return self.editor.dust(date, location, np.array(results))
                # 슬라이싱을 위해 numpy 어레이로 전송

    def _naver(self, query, selector):
        url = self.url['naver'] + urllib.parse.quote(query)
        out = bs4.BeautifulSoup(urlopen(Request(url)).read(), 'html.parser')
        return [s.contents[0] for s in out.select(selector)]
