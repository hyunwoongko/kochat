import urllib
from urllib.request import urlopen, Request

import bs4

from app.crawler_data import CrawlerData


class CrawlerAPI:

    def __init__(self):
        self.data = CrawlerData()

    def dust(self, location, date):
        if date in self.data.date['today']:
            pass

    def __connect_naver(self, query):
        url = self.data.url['naver'] + urllib.parse.quote(query)
        return bs4.BeautifulSoup(urlopen(Request(url)).read(), 'html.parser')

    def __connect_google(self, query):
        pass


def today_dust(location):
    enc_location = urllib.parse.quote(location + ' 오늘 날씨')
    url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location
    soup = bs4.BeautifulSoup(urlopen(Request(url)).read(), 'html.parser')

    dust_figure = soup.find('dl', class_='indicator')
    dust_figure = dust_figure.text.replace('㎍/㎥', '마이크로그램퍼미터 ').replace('ppm', '피피엠 ').split()
    del dust_figure[0]
    del dust_figure[2]
    del dust_figure[4]

    dust = '오늘 ' + location + '지역 미세먼지 정보를 알려드릴게요. ' + '오늘 ' + location + '지역의 미세먼지 상태는 ' + dust_figure[
        1] + ' 이고, 농도는 ' + dust_figure[0] + ', 초미세먼지 상태는 ' + dust_figure[3] + ' 이고, 농도는 ' + dust_figure[
               2] + ', 오존 상태는 ' + dust_figure[5] + ' 이고, 농도는 ' + dust_figure[4] + '입니다'

    if '나쁨' in dust:
        dust += ' 공기 상태가 안좋으니 마스크를 착용하세요'

    return dus
