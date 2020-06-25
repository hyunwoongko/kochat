from _crawler.answerer.base.answerer import Answerer


class WeatherAnswerer(Answerer):

    def comparison_with_yesterday_form(self, location: str, date: str, result: dict) -> str:
        """
        어제 온도와 비교하는 출력 포맷

        :param location: 지역
        :param date: 날짜
        :param result: 데이터 딕셔너리
        :return: 출력 메시지
        """

        msg = self.weather.format(location=location)
        msg += '{date} {location}지역은 섭씨 {temperature}도이며, {comparison}. {weather}' \
            .format(date=date, location=location,
                    temperature=result['temperature'],
                    comparison=result['comparison'],
                    weather=result['weather'])

        return msg

    def specific_date_form(self, location: str, date: str, result: dict) -> str:
        """
        특정 날짜 (오전/오후 구분 없는) 출력 포맷
        
        :param location: 지역
        :param date: 날짜
        :param result: 데이터 딕셔너리
        :return: 출력 메시지
        """

        msg = self.weather.format(location=location)
        msg += '{date} {location}지역은 섭씨 {temperature}도이며, {weather}' \
            .format(date=date, location=location,
                    temperature=result['temperature'],
                    weather=result['weather'])

        return msg

    def morning_afternoon_form(self, location: str, date: str, result: dict, josa: list) -> str:
        """
        오전-오후로 구성된 출력 포맷

        :param location: 지역
        :param date: 날짜
        :param josa: 조사 리스트
        :param result: 데이터 딕셔너리
        :return: 출력 메시지
        """

        msg = self.weather.format(location=location)
        msg += '{date} {location}지역은 오전에{j1} 섭씨 {t1}도이며, {w1} ' \
               '오후에{j2} 섭씨 {t2}도이며, {w2}' \
            .format(date=date, location=location,
                    j1=josa[0], j2=josa[1],
                    t1=result['morning_temperature'],
                    t2=result['afternoon_temperature'],
                    w1=result['morning_weather'],
                    w2=result['afternoon_weather'])

        return msg
