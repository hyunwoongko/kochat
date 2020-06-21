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

        self.num_component_when_success1 = 18
        self.num_component_when_success2 = 60
        self.date_coverage = self.date['today'] + \
                             self.date['tomorrow'] + \
                             self.date['after_tomorrow']

    def crawl(self, location, date):
        try:
            return self.crawl_debug(location, date)
        except:
            return self.answerer.sorry("해당 대기 오염정보는 알 수 없습니다.")

    def crawl_debug(self, location, date):
        if date in self.date_coverage:
            # 네이버 미세먼지 API 신버전으로 우선 탐색
            return self.__new_everyday(location, date)

        else:
            return self.answerer.sorry(
                ', '.join(self.date_coverage)
                + "의 대기오염 정보만 알 수 있어요!"
            )

    def __new_everyday(self, location, date):
        """
        네이버 미세먼지 신버전 (오늘/내일/모레) 검색
        실패시 구버전 검색 시도
        """
        result = self.searcher.new_everyday(location, date)

        if len(result) == self.num_component_when_success1:  # 성공시 반드시 18개의 상태가 리턴됨
            result, josa = self.editor.edit_new_everyday(location, date, result)
            return self.answerer.morning_afternoon_form(location, date, result, josa)
        else:
            # 실패시 네이버 미세먼지 구버전으로 검색 시도
            return self.__old_version(location, date)

    def __old_version(self, location, date):
        """
        앞전에 날짜를 체크했기 때문에 반드시 Date Coverage 안에 있음
        네이버 미세먼지 구버전 (오늘) or (내일/모레) 검색 시도
        """
        if date in self.date['today']:
            return self.__old_today(location, date)
        else:
            return self.__old_tomorrow(location, date)

    def __old_today(self, location, date):
        """
        성공시 메시지 출력, 실패시 실패 메시지 출력
        """
        result = self.searcher.old_today(location, date)

        if len(result) == self.num_component_when_success1:  # 성공시 반드시 18개의 상태가 리턴됨
            result, josa = self.editor.edit_old_today(location, date, result)
            return self.answerer.single_form(location, date, result, josa)
        else:
            return DustAnswerer().sorry()

    def __old_tomorrow(self, location, date):
        """
        성공시 메시지 출력, 실패시 실패 메시지 출력
        """
        result = self.searcher.old_tomorrow(location, date)

        if len(result) == self.num_component_when_success2:  # 성공시 반드시 60개의 상태가 리턴됨
            result, josa = self.editor.edit_old_tomorrow(location, date, result)
            return self.answerer.morning_afternoon_form(location, date, result, josa)
        else:
            return self.answerer.sorry()
