"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.base_scenario import BaseScenario
from _crawler.crawler.travel_crawler import TravelCrawler


class TravelScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.crawler = TravelCrawler()

    def check_entity(self, text, entity):
        check_dict = self._check_entity(
            text=text,
            entity=entity,
            dict_={
                'TRAVEL': [],
                'LOCATION': []
            }
        )

        LOCATION = check_dict['LOCATION']
        TRAVEL = check_dict['TRAVEL']
        return LOCATION, TRAVEL

    def __call__(self, text, entity):
        location, travel = self.check_entity(text, entity)
        travel = self._set_as_default(travel, '')
        location, travel = ' '.join(location), ' '.join(travel)

        if len(location) != 0:
            return {'intent': '여행지',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, travel),
                    'state': 'SUCCESS'}

        else:
            return {'intent': '먼지',
                    'entity': entity,
                    'answer': None,
                    'state': 'REQUIRE_LOCATION'}
