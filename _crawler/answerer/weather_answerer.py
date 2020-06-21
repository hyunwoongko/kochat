from _crawler.answerer.base.answerer import Answerer


class WeatherAnswerer(Answerer):

    def comparison_with_yesterday_form(self, location, date, result):
        weather, comparison, temperature = result[0], result[1], result[2]

        msg = self.weather.format(location=location)
        msg += '{date} {location}지역은 섭씨 {temperature}도이며, {comparison}. {weather}' \
            .format(date=date, location=location, temperature=temperature,
                    comparison=comparison, weather=weather)
        return msg

    def specific_date_form(self, location, date, result):
        weather, temperature = result[0], result[1]
        msg = self.weather.format(location=location)
        msg += '{date} {location}지역은 섭씨 {temperature}도이며, {weather}' \
            .format(date=date, location=location, temperature=temperature, weather=weather)

        return msg

    def morning_afternoon_form(self, location, date, result, josa):
        morning_weather, morning_temp = result[0], result[1]
        afternoon_weather, afternoon_temp = result[2], result[3]
        msg = self.weather.format(location=location)
        msg += '{date} {location}지역은 오전에{j1} 섭씨 {t1}도이며, {w1} ' \
               '오후에{j2} 섭씨 {t2}도이며, {w2}' \
            .format(date=date, location=location,
                    j1=josa[0], t1=morning_temp, w1=morning_weather,
                    j2=josa[1], t2=afternoon_temp, w2=afternoon_weather)

        return msg
