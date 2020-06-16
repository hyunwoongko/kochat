from crawler.base.base import Crawler
from crawler.core.answerer.dust_answerer import DustAnswerer
from crawler.core.editor.dust_editor import DustEditor
from crawler.core.searcher.dust_searcher import DustSearcher
from util.oop import singleton


@singleton
class DustCrawler(Crawler):

    def __init__(self):
        super().__init__()
        self.num_component_when_success = 18
        self.date_coverage = self.date['today'] + \
                             self.date['tomorrow'] + \
                             self.date['after']

    def request(self, date, location):
        """
        대답가능한 날짜 커버리지안에 없는 날짜를 물어봤으면 이에 관한 실패 메시지 출력
        """
        if date in self.date_coverage:
            return self._new_everyday(date, location)
        else:
            return DustAnswerer().sorry(
                ', '.join(self.date_coverage)
                + "의 대기오염 정보만 알 수 있어요!"
            )

    def _new_everyday(self, date, location):
        """
        네이버 미세먼지 신버전 (오늘/내일/모레) 검색
        실패시 구버전 검색 시도
        """
        result = DustSearcher().new_everyday(date, location)

        if len(result) == self.num_component_when_success:  # 성공시 반드시 18개의 상태가 리턴됨
            return DustEditor().new_everyday(date, location, result)
        else:
            return self._old_version(date, location)

    def _old_version(self, date, location):
        """
        앞전에 날짜를 체크했기 때문에 반드시 Date Coverage 안에 있음
        네이버 미세먼지 구버전 (오늘) or (내일/모레) 검색 시도
        """
        if date in self.date['today']:
            return self._old_today(date, location)
        else:
            return self._old_tomorrow(date, location)

    def _old_today(self, date, location):
        """
        성공시 메시지 출력, 실패시 실패 메시지 출력
        """
        result = DustSearcher().old_today(date, location)
        if len(result) == self.num_component_when_success:  # 성공시 반드시 18개의 상태가 리턴됨
            return DustEditor().old_today(date, location, result)
        else:
            return DustAnswerer().sorry()

    def _old_tomorrow(self, date, location):
        """
        성공시 메시지 출력, 실패시 실패 메시지 출력
        """
        result = DustSearcher().old_tomorrow(date, location)
        if len(result) == 18:
            return DustEditor().old_tomorrow(date, location, result)
        else:
            return DustAnswerer().sorry()
