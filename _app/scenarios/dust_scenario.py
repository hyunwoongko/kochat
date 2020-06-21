"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.base_scenario import BaseScenario
from _crawler.crawler.dust_crawler import DustCrawler


class DustScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.crawler = DustCrawler()

    def check_entity(self, text, entity):
        check_dict = self._check_entity(
            text=text,
            entity=entity,
            dict_={
                'DATE': [],
                'LOCATION': []
            }
        )

        LOCATION = check_dict['LOCATION']
        DATE = check_dict['DATE']
        return LOCATION, DATE

    def __call__(self, text, entity):
        location, date = self.check_entity(text, entity)
        date = self._set_as_default(date, '오늘')

        if len(location) != 0:
            return {'intent': '먼지',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, date),
                    'state': 'SUCCESS'}

        else:
            return {'intent': '먼지',
                    'entity': entity,
                    'answer': None,
                    'state': 'REQUIRE_LOCATION'}
