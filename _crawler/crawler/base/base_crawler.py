import itertools
import re

import numpy as np

from _crawler.decorators import crawler


@crawler
class BaseCrawler:

    def _untag(self, text: str) -> str:
        """
        html 태그를 제거합니다.
        
        :param text: html 태그를 제거할 문자열
        :return: html 태그가 제거된 문자열
        """

        return re.sub(re.compile('<.*?>'), '', str(text))

    def _flatten_list(self, list_: list) -> list:
        """
        n차원 리스트를 1차원으로 만듭니다.
        
        :param list_: n차원 리스트 
        :return: 1차원으로 변형된 리스트
        """

        np.array(list_)
        while len(np.array(list_).shape) != 1:
            list_ = list(itertools.chain.from_iterable(list_))
        return list_

    def _flatten_dicts(self, dict_: dict) -> dict:
        """
        딕셔너리의 value 리스트를 전부 1차원으로 만듭니다.

        before :
        dict = {
            key1 : [[val1, val2], [val3, val4], [val5, val6]]
            key2 : [[val1, val2], [val3, val4], [val5, val6]]
        }

        after :
        dict = {
            key1 : [val1, val2, val3, val4, val5, val6]
            key2 : [val1, val2, val3, val4, val5, val6]
        }

        :param dict_: n차원 리스트를 값으로 가지고 있는 딕셔너리
        :return: 1차원 리스트만 값으로 가지고 있는 딕셔너리
        """
        for k, v in dict_.items():
            dict_[k] = self._flatten_list(dict_[k])
        return dict_
