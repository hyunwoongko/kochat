"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from random import randint

from _crawler.searcher.base.searcher import Searcher


class TravelSearcher(Searcher):

    def __init__(self):
        super().__init__()
        self.data_dict = {
            'name': [],
            'tel': [],
            'context': [],
            'category': [],
            'address': [],
            'thumUrl': []}

    def _make_query(self, location, travel):
        query = ' '.join([location, travel])
        query += " 여행"
        return query

    def search_naver_map(self, location, travel):
        query = self._make_query(location, travel)
        result = self._ajax_json(url=self.url['naver_map'],
                                 query=query)

        if result is not None:
            result = result['result']['place']['list']
            random_selected_result = result[randint(0, len(result) - 1)]

            self.data_dict['name'].append(random_selected_result['name'])
            self.data_dict['context'].append(random_selected_result['context'])
            self.data_dict['category'].append(random_selected_result['category'])
            self.data_dict['address'].append(random_selected_result['address'])
            self.data_dict['thumUrl'].append(random_selected_result['thumUrl'])
            self.data_dict = self._flatten_dicts(self.data_dict)
            return self.data_dict

        else:
            return None
