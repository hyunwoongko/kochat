"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.base_scenario import BaseScenario
from _crawler.crawler.retaurant_crawler import RestaurantCrawler


class RestaurantScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.crawler = RestaurantCrawler()

    def check_entity(self, text, entity):
        check_dict = self._check_entity(
            text=text,
            entity=entity,
            dict_={
                'LOCATION': [],
                'RESTAURANT': []
            }
        )

        LOCATION = check_dict['LOCATION']
        RESTAURANT = check_dict['RESTAURANT']
        return LOCATION, RESTAURANT

    def __call__(self, text, entity):
        location, restaurant = self.check_entity(text, entity)
        restaurant = self._set_as_default(restaurant, '')
        location, date = ' '.join(location), ' '.join(restaurant)

        if len(location) != 0:
            return {'intent': '맛집',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, restaurant),
                    'state': 'SUCCESS'}

        else:
            return {'intent': '맛집',
                    'entity': entity,
                    'answer': None,
                    'state': 'REQUIRE_LOCATION'}
