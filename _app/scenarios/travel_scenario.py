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

    def check_entity(self, text: list, entity: list) -> tuple:
        """
        TRAVEL와 LOCATION에 해당하는 단어를 리스트에 추가합니다.

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: location 리스트, travel 리스트
        """

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

    def __call__(self, text: list, entity: list) -> dict:
        """
        location이 있다면 크롤링 수행
        location이 없다면 require_location 리턴

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: json 딕셔너리
        """

        location, travel = self.check_entity(text, entity)
        travel = self._set_as_default(travel, '')
        location, travel = ' '.join(location), ' '.join(travel)

        if len(location) != 0:
            return {'intent': 'travel',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, travel),
                    'state': 'SUCCESS'}

        else:
            return {'intent': 'travel',
                    'entity': entity,
                    'answer': None,
                    'state': 'REQUIRE_LOCATION'}
