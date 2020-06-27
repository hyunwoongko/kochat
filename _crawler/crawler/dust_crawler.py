<<<<<<< HEAD
from _crawler.answerer.dust_answerer import DustAnswerer
from _crawler.crawler.base.base_crawler import BaseCrawler
from _crawler.editor.dust_editor import DustEditor
from _crawler.searcher.dust_searcher import DustSearcher


class DustCrawler(BaseCrawler):

    def __init__(self):
        super().__init__()
        self.answerer = DustAnswerer()
        self.editor = DustEditor()
        self.searcher = DustSearcher()

        self.date_coverage = self.date['today'] + \
                             self.date['tomorrow'] + \
                             self.date['after']

    def crawl(self, location: str, date: str) -> str:
        """
        미세먼지를 크롤링합니다.
        (try-catch로 에러가 나지 않는 함수)

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        try:
            return self.crawl_debug(location, date)
        except Exception:
            return self.answerer.sorry(
                "해당 대기 오염정보는 알 수 없습니다."
            )

    def crawl_debug(self, location: str, date: str) -> str:
        """
        미세먼지를 크롤링합니다.
        (에러가 나는 디버깅용 함수)

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        if date in self.date_coverage:
            # 네이버 미세먼지 API 신버전으로 우선 탐색
            return self.__new_everyday(location, date)

        else:
            return self.answerer.sorry(
                ', '.join(self.date_coverage)
                + "의 대기오염 정보만 알 수 있어요!"
            )

    def __new_everyday(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        우선 신버전(오늘,내일,모레)으로 검색한 뒤  실패시 구버전 검색 시도

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        try:
            result = self.searcher.new_everyday(location, date)
            result, josa = self.editor.edit_morning_afternoon(location, date, result)
            return self.answerer.morning_afternoon_form(location, date, result, josa)
        except Exception:
            return self.__old_version(location, date)

    def __old_version(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        신버전에서 실패할시 구버전으로 검색을 시도함 (날짜에 따라 오늘, 내일/모레 뷰가 다름)

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        if date in self.date['today']:
            return self.__old_today(location, date)
        else:
            return self.__old_tomorrow(location, date)

    def __old_today(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        신버전에서 실패할시 구버전(오늘) 검색을 시도함.

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        result = self.searcher.old_today(location, date)
        result, josa = self.editor.edit_single(location, date, result)
        return self.answerer.single_form(location, date, result, josa)

    def __old_tomorrow(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        신버전에서 실패할시 구버전(내일/모레) 검색을 시도함.

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        result = self.searcher.old_tomorrow(location, date)
        result, josa = self.editor.edit_morning_afternoon(location, date, result)
        return self.answerer.morning_afternoon_form(location, date, result, josa)
=======
from _crawler.answerer.dust_answerer import DustAnswerer
from _crawler.crawler.base.crawler import Crawler
from _crawler.editor.dust_editor import DustEditor
from _crawler.searcher.dust_searcher import DustSearcher


class DustCrawler(Crawler):

    def __init__(self):
        super().__init__()
        self.answerer = DustAnswerer()
        self.editor = DustEditor()
        self.searcher = DustSearcher()

        self.date_coverage = self.date['today'] + \
                             self.date['tomorrow'] + \
                             self.date['after']

    def crawl(self, location: str, date: str) -> str:
        """
        미세먼지를 크롤링합니다.
        (try-catch로 에러가 나지 않는 함수)

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        try:
            return self.crawl_debug(location, date)
        except:
            return self.answerer.sorry(
                "해당 대기 오염정보는 알 수 없습니다."
            )

    def crawl_debug(self, location: str, date: str) -> str:
        """
        미세먼지를 크롤링합니다.
        (에러가 나는 디버깅용 함수)

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        if date in self.date_coverage:
            # 네이버 미세먼지 API 신버전으로 우선 탐색
            return self.__new_everyday(location, date)

        else:
            return self.answerer.sorry(
                ', '.join(self.date_coverage)
                + "의 대기오염 정보만 알 수 있어요!"
            )

    def __new_everyday(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        우선 신버전(오늘,내일,모레)으로 검색한 뒤  실패시 구버전 검색 시도

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        try:
            result = self.searcher.new_everyday(location, date)
            result, josa = self.editor.edit_morning_afternoon(location, date, result)
            return self.answerer.morning_afternoon_form(location, date, result, josa)
        except:
            return self.__old_version(location, date)

    def __old_version(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        신버전에서 실패할시 구버전으로 검색을 시도함 (날짜에 따라 오늘, 내일/모레 뷰가 다름)

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        if date in self.date['today']:
            return self.__old_today(location, date)
        else:
            return self.__old_tomorrow(location, date)

    def __old_today(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        신버전에서 실패할시 구버전(오늘) 검색을 시도함.

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        result = self.searcher.old_today(location, date)
        result, josa = self.editor.edit_single(location, date, result)
        return self.answerer.single_form(location, date, result, josa)

    def __old_tomorrow(self, location: str, date: str) -> str:
        """
        네이버 미세먼지는 신/구버전이 있는데,
        주요 도시는 신버전으로, 군/구/읍/면/동 등 시 이하의 행정구역은 구버전으로 구현되어있음.
        신버전에서 실패할시 구버전(내일/모레) 검색을 시도함.

        :param location: 지역
        :param date: 날짜
        :return: 해당지역 날짜 미세먼지
        """

        result = self.searcher.old_tomorrow(location, date)
        result, josa = self.editor.edit_morning_afternoon(location, date, result)
        return self.answerer.morning_afternoon_form(location, date, result, josa)
>>>>>>> 998bcd017cd44db5c996455ee9ee1193cb11520e
