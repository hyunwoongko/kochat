"""
@auther Hyunwoong
@since {6/20/2020}
@see : https://github.com/gusdnd852
"""
from urllib.request import Request, urlopen

import bs4

from _dialogue.searcher.base.base_searcher import BaseSearcher
from random import randint


class RestaurantSearcher(BaseSearcher):

    def __init__(self):
        self.CSS = {
            # 검색에 사용할 CSS 셀렉터들을 정의합니다.
            'names': '.info_area > .tit > .tit_inner > .name',
            'name': '.ct_box_area > .biz_name_area > strong.name',
            'phone_number': '.list_bizinfo > .list_item.list_item_biztel > .txt',
            'category': '.ct_box_area > .biz_name_area > span.category',
            'time': '.txt > .biztime_area.list_more_view > .biztime_row > .biztime > span.time',
            'address': '.ct_box_area > .bizinfo_area > .list_bizinfo > .list_item.list_item_address'
                       ' > .txt > .list_address > li > .addr',
        }

        self.data_dict = {
            # 데이터를 담을 딕셔너리 구조를 정의합니다.
            'name': [], 'phone_number': [],
            'category': [], 'time': [],
            'address': []
        }

    def _make_query(self, location: str, restaurant: str) -> str:
        """
        검색할 쿼리를 만듭니다.

        :param location: 지역
        :param restaurant: 맛집종류
        :return: "지역 맛집종류 맛집"으로 만들어진 쿼리
        """

        return ' '.join([location, restaurant, '맛집'])

    def naver_search(self, location: str, restaurant: str) -> dict:
        """
        1차적으로 네이버에서 맛집 리스트를 얻기 위해 맛집을 검색합니다.
        
        :param location: 지역
        :param restaurant: 맛집 종류
        :return:
        """

        query = self._make_query(location, restaurant)
        result = self._bs4_documents(self.url['naver'],
                                     selectors=[self.CSS['names']],
                                     query=query)

        result = [place.get('href') for place in result]
        url = result[randint(0, len(result) - 1)]
        # 랜덤으로 검색된 맛집 중 하나를 고르고
        # 내용을 자세히 보기 위해 플레이스로 이동

        result = self.__naver_place(url)
        return result

    def __naver_place(self, url: str) -> dict:
        """
        랜덤하게 선택된 맛집의 네이버 플레이스 url을 가져와서 접속합니다.
        이를 통해 해당 맛집의 세부 정보를 크롤링 할 수 있습니다.
        
        :param url: 네이버 플레이스 url
        :return: 데이터 딕셔너리
        """

        result = self._bs4_documents(url, selectors=[self.CSS['name'],
                                                     self.CSS['phone_number'],
                                                     self.CSS['category'],
                                                     self.CSS['time'],
                                                     self.CSS['address']])

        # 플레이스에서 뽑힌 값중 필요한 값들 골라서 저장
        for r in result:
            if 'name' in str(r):
                self.data_dict['name'].append(self._untag(str(r)))
            elif 'txt' in str(r):
                self.data_dict['phone_number'].append(self._untag(str(r)))
            elif 'category' in str(r):
                self.data_dict['category'].append(self._untag(str(r)))
            elif '<span class="time">' in str(r):
                self.data_dict['time'].append(self._untag(str(r)))
            elif 'addr' in str(r):
                self.data_dict['address'].append(self._untag(str(r)))

        return self.data_dict
