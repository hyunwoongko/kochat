from _crawler.editor.base.editor import Editor
import re


class WeatherEditor(Editor):

    def edit_today(self, result):
        weather = result[0][0]
        temperature = result[5][0]
        compare_with_yesterday = weather.split(',')[1].strip()
        weather = weather.split(',')[0].strip()

        weather = self.weather[weather]
        result = [weather, compare_with_yesterday, temperature]
        return result

    def edit_tomorrow(self, result):
        morning_weather = result[1][0]
        afternoon_weather = result[2][0]
        morning_temperature = result[6][0]
        afternoon_temperature = result[7][0]

        morning_weather = self.weather[morning_weather]
        afternoon_weather = self.weather[afternoon_weather]
        josa = self.enumerate_josa('는', '도', [morning_weather, afternoon_weather])
        result = [morning_weather, morning_temperature, afternoon_weather, afternoon_temperature]
        return result, josa

    def edit_after_tomorrow(self, result):
        morning_weather = result[3][0]
        afternoon_weather = result[4][0]
        morning_temperature = result[8][0]
        afternoon_temperature = result[9][0]

        morning_weather = self.weather[morning_weather]
        afternoon_weather = self.weather[afternoon_weather]
        josa = self.enumerate_josa('는', '도', [morning_weather, afternoon_weather])
        result = [morning_weather, morning_temperature, afternoon_weather, afternoon_temperature]
        return result, josa

    def edit_specific(self, result):
        weather = result[0][0]
        temperature = result[1][0]
        weather = re.sub(' ', '', weather)
        weather = self.weather[weather]
        result = [weather, temperature]
        return result
