<<<<<<< HEAD
"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.base_scenario import BaseScenario
from _crawler.crawler.weather_crawler import WeatherCrawler


class WeatherScenario(BaseScenario):

    def __init__(self):
        """
        날씨 시나리오 클래스입니다.
        """

        super().__init__()
        self.crawler = WeatherCrawler()

    def check_entity(self, text: list, entity: list) -> tuple:
        """
        DATE와 LOCATION에 해당하는 단어를 리스트에 추가합니다.

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: location 리스트, date 리스트
        """

        check_dict = self._check_entity(
            text=text,
            entity=entity,
            dict_={
                'LOCATION': [],
                'DATE': []
            }
        )

        LOCATION = check_dict['LOCATION']
        DATE = check_dict['DATE']
        return LOCATION, DATE

    def __call__(self, text: list, entity: list) -> dict:
        """
        location이 있다면 크롤링 수행
        location이 없다면 require_location 리턴

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: json 딕셔너리
        """

        location, date = self.check_entity(text, entity)
        date = self._set_as_default(date, '오늘')
        location, date = ' '.join(location), ' '.join(date)

        if len(location) != 0:
            return {'intent': '날씨',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, date),
                    'state': 'SUCCESS'}

        else:
            return {'intent': '날씨',
                    'entity': entity,
                    'answer': None,
                    'state': 'REQUIRE_LOCATION'}
=======
"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.base_scenario import BaseScenario
from _crawler.crawler.weather_crawler import WeatherCrawler


class WeatherScenario(BaseScenario):

    def __init__(self):
        """
        날씨 시나리오 클래스입니다.
        """

        super().__init__()
        self.crawler = WeatherCrawler()

    def check_entity(self, text: list, entity: list) -> tuple:
        """
        DATE와 LOCATION에 해당하는 단어를 리스트에 추가합니다.

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: location 리스트, date 리스트
        """

        check_dict = self._check_entity(
            text=text,
            entity=entity,
            dict_={
                'LOCATION': [],
                'DATE': []
            }
        )

        LOCATION = check_dict['LOCATION']
        DATE = check_dict['DATE']
        return LOCATION, DATE

    def __call__(self, text: list, entity: list) -> dict:
        """
        location이 있다면 크롤링 수행
        location이 없다면 require_location 리턴

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: json 딕셔너리
        """

        location, date = self.check_entity(text, entity)
        date = self._set_as_default(date, '오늘')
        location, date = ' '.join(location), ' '.join(date)

        if len(location) != 0:
            return {'intent': '날씨',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, date),
                    'state': 'SUCCESS'}

        else:
            return {'intent': '날씨',
                    'entity': entity,
                    'answer': None,
                    'state': 'REQUIRE_LOCATION'}
>>>>>>> 998bcd017cd44db5c996455ee9ee1193cb11520e
