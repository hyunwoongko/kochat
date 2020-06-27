from _dialogue.searcher.base.base_searcher import BaseSearcher


class WeatherSearcher(BaseSearcher):

    def __init__(self):
        self.CSS = {
            # 검색에 사용할 CSS 셀렉터들을 정의합니다.
            'naver_weather': '.info_data > .info_list > li > .cast_txt',
            'naver_temperature': '.info_temperature > .todaytemp',
            'google_weather': '#wob_dcp > #wob_dc',
            'google_temperature': '#wob_tm'
        }

        self.data_dict = {
            # 데이터를 담을 딕셔너리 구조를 정의합니다.
            'today_weather': None,
            'tomorrow_morning_weather': None,
            'tomorrow_afternoon_weather': None,
            'after_morning_weather': None,
            'after_afternoon_weather': None,
            'specific_weather': None,
            'today_temperature': None,
            'tomorrow_morning_temperature': None,
            'tomorrow_afternoon_temperature': None,
            'after_morning_temperature': None,
            'after_afternoon_temperature': None,
            'specific_temperature': None,
        }

    def _make_query(self, location: str, date: str) -> str:
        """
        검색할 쿼리를 만듭니다.
        
        :param location: 지역
        :param date: 날짜
        :return: "지역 날짜 날씨"로 만들어진 쿼리
        """

        return ' '.join([location, date, '날씨'])

    def naver_search(self, location: str) -> dict:
        """
        네이버를 이용해 날씨를 검색합니다.

        :param location: 지역
        :return: 크롤링된 내용
        """

        query = self._make_query('오늘', location)  # 한번 서치에 전부 가져옴
        result = self._bs4_contents(self.url['naver'],
                                    selectors=[self.CSS['naver_weather'],
                                               self.CSS['naver_temperature']],
                                    query=query)

        i = 0
        for k in self.data_dict.keys():
            if 'specific' not in k:
                # specific 빼고 전부 담음
                self.data_dict[k] = result[i][0]
                i += 1

        return self.data_dict

    def google_search(self, location: str, date: str) -> dict:
        """
        구글을 이용해 날씨를 검색합니다.

        :param location: 지역
        :param date: 날짜
        :return: 크롤링된 내용
        """

        query = self._make_query(location, date)  # 날짜마다 따로 가져와야함
        result = self._bs4_contents(self.url['google'],
                                    selectors=[self.CSS['google_weather'],
                                               self.CSS['google_temperature']],
                                    query=query)

        self.data_dict['specific_weather'] = result[0][0]
        self.data_dict['specific_temperature'] = result[1][0]
        # specific만 담음

        return self.data_dict
