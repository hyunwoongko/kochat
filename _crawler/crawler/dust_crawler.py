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

        self.num_component_when_success = 18
        self.num_component_when_success_ = 60
        self.date_coverage = self.date['today'] + \
                             self.date['tomorrow'] + \
                             self.date['after']

    def crawl(self, date, location):
        """
        대답가능한 날짜 커버리지안에 없는 날짜를 물어봤으면 이에 관한 실패 메시지 출력
        """
        if date in self.date_coverage:
            # 네이버 미세먼지 API 신버전으로 우선 탐색
            return self.__new_everyday(date, location)

        else:
            return self.answerer.sorry(
                ', '.join(self.date_coverage)
                + "의 대기오염 정보만 알 수 있어요!"
            )

    def __new_everyday(self, date, location):
        """
        네이버 미세먼지 신버전 (오늘/내일/모레) 검색
        실패시 구버전 검색 시도
        """
        result = self.searcher.new_everyday(date, location)

        if len(result) == self.num_component_when_success:
            result, josa = self.editor.new_everyday(date, location, result)
            return self.answerer.new_everyday(date, location, result, josa)
        else:
            # 실패시 네이버 미세먼지 구버전으로 검색 시도
            return self.__old_version(date, location)

    def __old_version(self, date, location):
        """
        앞전에 날짜를 체크했기 때문에 반드시 Date Coverage 안에 있음
        네이버 미세먼지 구버전 (오늘) or (내일/모레) 검색 시도
        """
        if date in self.date['today']:
            return self.__old_today(date, location)
        else:
            return self.__old_tomorrow(date, location)

    def __old_today(self, date, location):
        """
        성공시 메시지 출력, 실패시 실패 메시지 출력
        """
        result = self.searcher.old_today(date, location)

        if len(result) == self.num_component_when_success:  # 성공시 반드시 18개의 상태가 리턴됨
            result, josa = self.editor.old_today(date, location, result)
            return self.answerer.old_today(date, location, result, josa)
        else:
            return DustAnswerer().sorry()

    def __old_tomorrow(self, date, location):
        """
        성공시 메시지 출력, 실패시 실패 메시지 출력
        """
        result = self.searcher.old_tomorrow(date, location)

        if len(result) == self.num_component_when_success_:  # 성공시 반드시 60개의 상태가 리턴됨
            result, josa = self.editor.old_tomorrow(date, location, result)
            return self.answerer.old_tomorrow(date, location, result, josa)
        else:
            return self.answerer.sorry()
