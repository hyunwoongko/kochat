from _crawler.searcher.base.searcher import Searcher


class DustSearcher(Searcher):
    def __init__(self):
        self.CSS = {
            # 검색에 사용할 CSS 셀렉터들을 정의합니다.
            'new_everyday': '.on.now > em',
            'old_today': 'div.all_state > ul.state_list > li > span.state_info',
            'old_tomorrow': 'div.air_detail > div.tb_scroll > table > tbody > tr > td'
        }

        self.new_data_dict = {
            # 미세먼지 데이터 목록
            'today_morning_fine_dust': None,
            'today_afternoon_fine_dust': None,
            'tomorrow_morning_fine_dust': None,
            'tomorrow_afternoon_fine_dust': None,
            'after_morning_fine_dust': None,
            'after_afternoon_fine_dust': None,

            # 초미세먼지 데이터 목록
            'today_morning_ultra_dust': None,
            'today_afternoon_ultra_dust': None,
            'tomorrow_morning_ultra_dust': None,
            'tomorrow_afternoon_ultra_dust': None,
            'after_morning_ultra_dust': None,
            'after_afternoon_ultra_dust': None,

            # 오존 데이터 목록
            'today_morning_ozon': None,
            'today_afternoon_ozon': None,
            'tomorrow_morning_ozon': None,
            'tomorrow_afternoon_ozon': None,
            'after_morning_ozon': None,
            'after_afternoon_ozon': None,
        }

        self.old_data_dict = {
            # 오늘 데이터 목록
            'today_fine_dust': None,
            'today_ultra_dust': None,
            'today_ozon': None,

            # 미세먼지 데이터 목록
            'tomorrow_morning_fine_dust': None,
            'tomorrow_afternoon_fine_dust': None,
            'after_morning_fine_dust': None,
            'after_afternoon_fine_dust': None,

            # 초미세먼지 데이터 목록
            'tomorrow_morning_ultra_dust': None,
            'tomorrow_afternoon_ultra_dust': None,
            'after_morning_ultra_dust': None,
            'after_afternoon_ultra_dust': None,

            # 오존 데이터 목록
            'tomorrow_morning_ozon': None,
            'tomorrow_afternoon_ozon': None,
            'after_morning_ozon': None,
            'after_afternoon_ozon': None,
        }

    def _make_query(self, location: str, date: str) -> list:
        """
        검색할 쿼리를 만듭니다.
        
        :param location: 지역
        :param date: 날짜
        :return: ["지역 날짜 미세먼지", "지역 날짜 초미세먼지", "지역 날짜 오존", ...]
        """

        return [' '.join([location, date] + [i])
                for i in self.kinds['dust']]

    def new_everyday(self, location: str, date: str) -> dict:
        """
        네이버 미세먼지는 신버전과 구버전으로 나뉘는데,
        신버전이 적용된 지역은 신버전으로 검색해서
        데이터 딕셔너리를 반환합니다.

        :param location: 지역
        :param date: 날짜
        :return: 데이터 딕셔너리
        """

        query = self._make_query(location, date)
        results = [self._bs4_contents(self.url['naver'],
                                      selectors=[self.CSS['new_everyday']],
                                      query=q) for q in query]

        for result in results:
            if len(result) == 0:
                # old version인 경우 ozon 리스트 길이 0으로 나옴
                raise Exception('old version 도시입니다.')

            while len(result) != 6:
                # 케이웨더 업데이트 안돼서 데이터 안나올때
                result.append(['데이터없음'])

        for i, kind in enumerate(['fine_dust', 'ultra_dust', 'ozon']):
            j = 0

            for date in ['today_', 'tomorrow_', 'after_']:
                for time in ['morning_', 'afternoon_']:
                    dict_key = date + time + kind
                    if kind == 'fine_dust':
                        self.new_data_dict[dict_key] = results[i][j][0]
                    elif kind == 'ultra_dust':
                        self.new_data_dict[dict_key] = results[i][j][0]
                    else:
                        self.new_data_dict[dict_key] = results[i][j][0]
                    j += 1

        return self.new_data_dict

    def old_today(self, location: str, date: str) -> dict:
        """
        네이버 미세먼지는 신버전과 구버전으로 나뉘는데,
        신버전이 미적용된 지역은 구버전으로 오늘 정보를 검색해서
        데이터 딕셔너리를 반환합니다.

        :param location: 지역
        :param date: 날짜
        :return: 데이터 딕셔너리
        """

        query = self._make_query(location, date)
        results = [self._bs4_contents(self.url['naver'], selectors=[self.CSS['old_today']], query=q)
                   for q in query]

        fine_dust = [result[0].strip() for result in results[0]]
        # fine dust 빼고 다 나옴 [ultra, ozon, ...]
        ultra_dust = [result[0].strip() for result in results[1]]
        # ultra dust 빼고 다 나옴 [fine, ozon, ...]

        self.old_data_dict['today_fine_dust'] = ultra_dust[0]
        self.old_data_dict['today_ultra_dust'] = fine_dust[0]
        self.old_data_dict['today_ozon'] = fine_dust[1]

        return self.old_data_dict

    def old_tomorrow(self, location: str, date: str) -> dict:
        """
        네이버 미세먼지는 신버전과 구버전으로 나뉘는데,
        신버전이 미적용된 지역은 구버전으로 내일/모레 정보를 검색해서
        데이터 딕셔너리를 반환합니다.

        :param location: 지역
        :param date: 날짜
        :return: 데이터 딕셔너리
        """

        query = self._make_query(location, date)[0]
        # 미세먼지만 검색해도 다 나옴 (1번만 검색하기)

        result = self._bs4_contents(self.url['naver'],
                                    selectors=[self.CSS['old_tomorrow']],
                                    query=query)

        result = [self._untag(r[0]) for r in result]

        i = 0
        for kind in ['fine_dust', 'ultra_dust', 'ozon']:
            for date in ['tomorrow_', 'after_']:
                for time in ['morning_', 'afternoon_']:
                    dict_key = date + time + kind
                    self.old_data_dict[dict_key] = result[i]
                    i += 1

        return self.old_data_dict
