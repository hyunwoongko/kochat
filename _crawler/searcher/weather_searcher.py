from _crawler.searcher.base.searcher import Searcher


class WeatherSearcher(Searcher):
    CSS = {
        'naver_weather': '.info_data > .info_list > li > .cast_txt',
        'naver_temperature': '.info_temperature > .todaytemp',
        'google_weather': '#wob_dcp > #wob_dc',
        'google_temperature': '#wob_tm'
    }

    def _make_query(self, location, date):
        return ' '.join([location, date, '날씨'])

    def naver_search(self, location):
        query = self._make_query('오늘', location)  # 한번 서치에 전부 가져옴
        result = self._bs4_contents(self.url['naver'],
                                    selectors=[self.CSS['naver_weather'],
                                                  self.CSS['naver_temperature']],
                                    query=query)

        return result

    def google_search(self, location, date):
        query = self._make_query(location, date)  # 날짜마다 따로 가져와야함
        result = self._bs4_contents(self.url['google'],
                                    selectors=[self.CSS['google_weather'],
                                                  self.CSS['google_temperature']],
                                    query=query)

        return result
