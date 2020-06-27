<<<<<<< HEAD
"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.base_scenario import BaseScenario
from _crawler.crawler.retaurant_crawler import RestaurantCrawler


class RestaurantScenario(BaseScenario):

    def __init__(self):
        """
        맛집 시나리오
        """
        
        super().__init__()
        self.crawler = RestaurantCrawler()

    def check_entity(self, text: list, entity: list) -> tuple:
        """
        RESTAURANT LOCATION에 해당하는 단어를 리스트에 추가합니다.

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: location 리스트, restaurant 리스트
        """

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

    def __call__(self, text: list, entity: list) -> dict:
        """
        location이 있다면 크롤링 수행
        location이 없다면 require_location 리턴

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: json 딕셔너리
        """

        location, restaurant = self.check_entity(text, entity)
        restaurant = self._set_as_default(restaurant, '')
        location, restaurant = ' '.join(location), ' '.join(restaurant)

        if len(location) != 0:
            return {'intent': 'restaurant',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, restaurant),
                    'state': 'SUCCESS'}

        else:
            return {'intent': 'restaurant',
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
from _crawler.crawler.retaurant_crawler import RestaurantCrawler


class RestaurantScenario(BaseScenario):

    def __init__(self):
        """
        맛집 시나리오
        """
        
        super().__init__()
        self.crawler = RestaurantCrawler()

    def check_entity(self, text: list, entity: list) -> tuple:
        """
        RESTAURANT LOCATION에 해당하는 단어를 리스트에 추가합니다.

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: location 리스트, restaurant 리스트
        """

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

    def __call__(self, text: list, entity: list) -> dict:
        """
        location이 있다면 크롤링 수행
        location이 없다면 require_location 리턴

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :return: json 딕셔너리
        """

        location, restaurant = self.check_entity(text, entity)
        restaurant = self._set_as_default(restaurant, '')
        location, restaurant = ' '.join(location), ' '.join(restaurant)

        if len(location) != 0:
            return {'intent': 'restaurant',
                    'entity': entity,
                    'answer': self.crawler.crawl(location, restaurant),
                    'state': 'SUCCESS'}

        else:
            return {'intent': 'restaurant',
                    'entity': entity,
                    'answer': None,
                    'state': 'REQUIRE_LOCATION'}
>>>>>>> 998bcd017cd44db5c996455ee9ee1193cb11520e
