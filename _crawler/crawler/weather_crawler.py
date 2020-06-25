from _crawler.answerer.weather_answerer import WeatherAnswerer
from _crawler.crawler.base.crawler import Crawler
from _crawler.editor.weather_editor import WeatherEditor
from _crawler.searcher.weather_searcher import WeatherSearcher


class WeatherCrawler(Crawler):

    def __init__(self):
        self.searcher = WeatherSearcher()
        self.editor = WeatherEditor()
        self.answerer = WeatherAnswerer()

    def crawl(self, location: str, date: str) -> str:
        """
        날씨를 크롤링합니다.
        (try-catch로 에러가 나지 않는 함수)

        :param location: 지역
        :param date: 날짜
        :return: 만들어진진 문장
       """

        try:
            return self.crawl_debug(location, date)
        except:
            return self.answerer.sorry(
                '그 날씨는 알 수가 없어요.'
            )

    def crawl_debug(self, location: str, date: str) -> str:
        """
        날씨를 크롤링합니다.
        (에러가 나는 디버깅용 함수)

        :param location: 지역
        :param date: 날짜
        :return: 만들어진진 문장
        """

        if date in self.date['today']:
            return self.__today(location)
        elif date in self.date['tomorrow']:
            return self.__tomorrow(location)
        elif date in self.date['after']:
            return self.__after(location)
        elif date in self.date['specific']:
            return self.__specific(location, date)
        else:
            try:
                return self.__specific(location, date)
            except:
                return self.answerer.sorry(
                    '그 때의 날씨는 알 수가 없어요.'
                )

    def __today(self, location: str) -> str:
        """
        오늘 날씨를 검색하고 조합합니다.

        :param location: 지역
        :return: 오늘 날씨
        """

        result = self.searcher.naver_search(location)
        result = self.editor.edit_today(result)
        return self.answerer.comparison_with_yesterday_form(location, "오늘", result)

    def __tomorrow(self, location: str) -> str:
        """
        내일 날씨를 검색하고 조합합니다.

        :param location: 지역
        :return: 내일 날씨
        """

        result = self.searcher.naver_search(location)
        result, josa = self.editor.edit_tomorrow(result)
        return self.answerer.morning_afternoon_form(location, "내일", result, josa)

    def __after(self, location: str) -> str:
        """
        모네 날씨를 검색하고 조합합니다.

        :param location: 지역
        :return: 모레 날씨
        """

        result = self.searcher.naver_search(location)
        result, josa = self.editor.edit_after(result)
        return self.answerer.morning_afternoon_form(location, "모레", result, josa)

    def __specific(self, location: str, date: str) -> str:
        """
        특정 날짜 (e.g. 수요일, 6월 20일 등)의
        날씨를 검색하고 조합합니다.

        :param location: 지역
        :return: 오늘 날씨
        """

        result = self.searcher.google_search(location, date)
        result = self.editor.edit_specific(result)
        return self.answerer.specific_date_form(location, date, result)
