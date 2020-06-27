<<<<<<< HEAD
"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.answerer.travel_answerer import TravelAnswerer
from _crawler.crawler.base.base_crawler import BaseCrawler
from _crawler.editor.travel_editor import TravelEditor
from _crawler.searcher.travel_searcher import TravelSearcher


class TravelCrawler(BaseCrawler):

    def __init__(self):
        self.searcher = TravelSearcher()
        self.editor = TravelEditor()
        self.answerer = TravelAnswerer()

    def crawl(self, location: str, travel: str) -> str:
        """
        여행지를 크롤링합니다.
        (try-catch로 에러가 나지 않는 함수)

        :param location: 지역
        :param travel: 여행지
        :return: 해당지역 여행지
        """

        try:
            return self.crawl_debug(location, travel)
        except Exception:
            return self.answerer.sorry(
                "해당 여행지는 알 수 없습니다."
            )

    def crawl_debug(self, location: str, travel: str) -> str:
        """
        여행지를 크롤링합니다.
        (에러가 나는 디버깅용 함수)

        :param location: 지역
        :param travel: 여행지
        :return: 해당지역 여행지
        """

        result = self.searcher.search_naver_map(location, travel)
        result = self.editor.edit_travel(location, travel, result)
        result = self.answerer.travel_form(location, travel, result)
        return result
=======
"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.answerer.travel_answerer import TravelAnswerer
from _crawler.crawler.base.crawler import Crawler
from _crawler.editor.travel_editor import TravelEditor
from _crawler.searcher.travel_searcher import TravelSearcher


class TravelCrawler(Crawler):

    def __init__(self):
        self.searcher = TravelSearcher()
        self.editor = TravelEditor()
        self.answerer = TravelAnswerer()

    def crawl(self, location: str, travel: str) -> str:
        """
        여행지를 크롤링합니다.
        (try-catch로 에러가 나지 않는 함수)

        :param location: 지역
        :param travel: 여행지
        :return: 해당지역 여행지
        """

        try:
            return self.crawl_debug(location, travel)
        except:
            return self.answerer.sorry(
                "해당 여행지는 알 수 없습니다."
            )

    def crawl_debug(self, location: str, travel: str) -> str:
        """
        여행지를 크롤링합니다.
        (에러가 나는 디버깅용 함수)

        :param location: 지역
        :param travel: 여행지
        :return: 해당지역 여행지
        """

        result = self.searcher.search_naver_map(location, travel)
        result = self.editor.edit_travel(location, travel, result)
        result = self.answerer.travel_form(location, travel, result)
        return result
>>>>>>> 998bcd017cd44db5c996455ee9ee1193cb11520e
