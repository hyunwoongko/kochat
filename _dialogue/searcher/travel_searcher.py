"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from random import randint

from _dialogue.searcher.base.base_searcher import BaseSearcher


class TravelSearcher(BaseSearcher):

    def __init__(self):
        self.data_dict = {
            # 데이터를 담을 딕셔너리 구조를 정의합니다.
            'name': [], 'tel': [],
            'context': [], 'category': [],
            'address': [], 'thumUrl': []
        }

    def _make_query(self, location: str, travel: str) -> str:
        """
        검색할 쿼리를 만듭니다.

        :param location: 지역
        :param travel: 여행지
        :return: "지역 여행지 여행"으로 만들어진 쿼리
        """

        query = ' '.join([location, travel])
        query += " 여행"
        return query

    def search_naver_map(self, location: str, travel: str) -> dict:
        """
        네이버 지도 API 에서 지역과 여행지를 검색합니다.

        :param location: 지역
        :param travel: 여행지
        :return: 사용할 내용만 json에서 뽑아서 dictionary로 만듬.
        """

        query = self._make_query(location, travel)
        result = self._json(url=self.url['naver_map'],
                            query=query)

        result = result['result']['place']['list']
        random_result = result[max(randint(0, len(result) - 1), 3)]
        # 네이버 지도 검색 결과 중에서 랜덤으로 하나 뽑음
        # 최대치는 3번째 칸에 출력된 결과 까지이며, 너무 뒷쪽 결과는 출력하지 않음

        self.data_dict['name'].append(random_result['name'])
        self.data_dict['context'].append(random_result['context'])
        self.data_dict['category'].append(random_result['category'])
        self.data_dict['address'].append(random_result['address'])
        self.data_dict['thumUrl'].append(random_result['thumUrl'])
        self.data_dict = self._flatten_dicts(self.data_dict)
        return self.data_dict
