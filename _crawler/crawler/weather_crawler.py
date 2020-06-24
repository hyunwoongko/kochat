from _crawler.answerer.weather_answerer import WeatherAnswerer
from _crawler.crawler.base.crawler import Crawler
from _crawler.editor.weather_editor import WeatherEditor
from _crawler.searcher.weather_searcher import WeatherSearcher


class WeatherCrawler(Crawler):

    def __init__(self):
        self.searcher = WeatherSearcher()
        self.editor = WeatherEditor()
        self.answerer = WeatherAnswerer()

    def crawl(self, location, date):
        try:
            return self.crawl_debug(location, date)
        except:
            return self.answerer.sorry(
                '그 날씨는 알 수가 없어요.'
            )

    def crawl_debug(self, location, date):
        if date in self.date['today']:
            return self.__today(location)
        elif date in self.date['tomorrow']:
            return self.__tomorrow(location)
        elif date in self.date['after_tomorrow']:
            return self.__after_tomorrow(location)
        elif date in self.date['specific']:
            return self.__specific(location, date)
        else:
            try:
                return self.__specific(location, date)
            except:
                return self.answerer.sorry(
                    '그 때의 날씨는 알 수가 없어요.'
                )

    def __today(self, location):
        result = self.searcher.naver_search(location)
        result = self.editor.edit_today(result)
        return self.answerer.comparison_with_yesterday_form("오늘", location, result)

    def __tomorrow(self, location):
        result = self.searcher.naver_search(location)
        result, josa = self.editor.edit_tomorrow(result)
        return self.answerer.morning_afternoon_form("내일", location, result, josa)

    def __after_tomorrow(self, location):
        result = self.searcher.naver_search(location)
        result, josa = self.editor.edit_after_tomorrow(result)
        return self.answerer.morning_afternoon_form("모레", location, result, josa)

    def __specific(self, location, date):
        result = self.searcher.google_search(location, date)
        result = self.editor.edit_specific(result)
        return self.answerer.specific_date_form(location, date, result)

    def __this_week(self, location):
        week_date = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        result = [self.__specific(location, date) for date in week_date]
        return ' '.join(result)
