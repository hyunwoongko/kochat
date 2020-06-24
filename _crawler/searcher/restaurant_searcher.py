"""
@auther Hyunwoong
@since {6/20/2020}
@see : https://github.com/gusdnd852
"""
from urllib.request import Request, urlopen

import bs4

from _crawler.searcher.base.searcher import Searcher
from random import randint


class RestaurantSearcher(Searcher):
    CSS = {
        # 네이버에서 처음 검색할때
        'names': '.info_area > .tit > .tit_inner > .name[href]',

        # 검색해서 네이버 플레이스 접속해서 정보 다 긁어 옴
        'name': '.ct_box_area > .biz_name_area > strong.name',
        'phone_number': '.list_bizinfo > .list_item.list_item_biztel > .txt',
        'category': '.ct_box_area > .biz_name_area > span.category',
        'time': '.txt > .biztime_area.list_more_view > .biztime_row > .biztime > span.time',
        'address': '.ct_box_area > .bizinfo_area > .list_bizinfo > .list_item.list_item_address'
                   ' > .txt > .list_address > li > .addr',
    }

    def __init__(self):
        super().__init__()
        self.data_dict = {
            'name': [],
            'phone_number': [],
            'category': [],
            'time': [],
            'address': []}

    def _make_query(self, location, restaurant):
        return ' '.join([location, restaurant, '맛집'])

    def naver_search(self, location, restaurant):
        query = self._make_query(location, restaurant)
        result = self._bs4_documents(self.url['naver'],
                                     selectors=[self.CSS['names']],
                                     query=query)

        result = [place.get('href') for place in result]
        url = result[randint(0, len(result) - 1)]
        result = self.__naver_place(url)
        return result

    def __naver_place(self, url):
        result = self._bs4_documents(url, selectors=[self.CSS['name'],
                                                     self.CSS['phone_number'],
                                                     self.CSS['category'],
                                                     self.CSS['time'],
                                                     self.CSS['address']])

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
