import urllib
from urllib.request import urlopen, Request

import bs4

metropolitans = [
    '서울', '서울시', '서울특별시',
    '대구', '대구시', '대구광역시',
    '대전', '대전시', '대전광역시',
    '부산', '부산시', '부산광역시',
    '인천', '인천시', '인천광역시',
    '광주', '광주시', '광주광역시',
    '울산', '울산시', '울산광역시',
    '제주', '제주도',
]


def today_dust(location):
    enc_location = urllib.parse.quote(location + ' 오늘 날씨')
    url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location
    locations = location.split(' ')
    req = Request(url)
    page = urlopen(req)
    html = page.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')

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

    return dust


def metropolitan(day, location):
    dust = day + ' ' + location + '의 미세먼지 정보를 알려드릴게요 '
    enc_location = urllib.parse.quote(location + '+ ' + day + ' 미세먼지')
    url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location
    req = Request(url)
    page = urlopen(req)
    html = page.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    dust_soup = soup.find_all('dl')
    dust_morn = dust_soup[6].text.split()[1]
    dust_noon = dust_soup[7].text.split()[1]
    dust += ', ' + day + ' ' + '오전 미세먼지 상태는 ' + dust_morn + ', 오후 상태는 ' + dust_noon
    enc_location = urllib.parse.quote(location + '+ ' + day + ' 초미세먼지')
    url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location
    req = Request(url)
    page = urlopen(req)
    html = page.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    dust_soup = soup.find_all('dl')
    dust_morn = dust_soup[6].text.split()[1]
    dust_noon = dust_soup[7].text.split()[1]
    dust += ', ' + day + ' 오전 미세먼지 상태는 ' + dust_morn + ', 오후 상태는 ' + dust_noon

    enc_location = urllib.parse.quote(location + '+ ' + day + ' 오존')
    url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location
    req = Request(url)
    page = urlopen(req)
    html = page.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    dust_soup = soup.find_all('dl')
    ozone_morn = dust_soup[6].text.split()[1]
    ozone_noon = dust_soup[7].text.split()[1]
    dust += ', ' + day + ' 오전 오존 상태는 ' + ozone_morn + ', 오후 상태는 ' + ozone_noon + '입니다'

    if '나쁨' in dust:
        dust += ' 공기 상태가 나쁘니 마스크를 착용하세요'

    return dust


def tomorrow_dust(location):
    if len(location.split()) == 1 and location in metropolitans:
        dust = metropolitan('내일', location)
    else:
        enc_location = urllib.parse.quote(location + '+ 내일 미세먼지')
        url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location

        req = Request(url)
        page = urlopen(req)
        html = page.read()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        dust_figure = soup.find_all('tbody')[2].text.split()
        dust_figure.remove('미세먼지')
        dust_figure.remove('초미세먼지')
        dust_figure.remove('오존')
        dust_figure.remove('자외선')
        dust_figure.remove('황사')

        dust = '내일 ' + location + '의 미세먼지 정보를 알려드릴게요 '
        dust_morn = dust_figure[0]
        dust_noon = dust_figure[1]
        dust += ', 내일 오전 미세먼지 상태는 ' + dust_morn + ', 오후 상태는 ' + dust_noon
        supdust_morn = dust_figure[4]
        supdust_noon = dust_figure[5]
        dust += ', 내일 오전 초미세먼지 상태는 ' + supdust_morn + ', 오후 상태는 ' + supdust_noon
        ozone_morn = dust_figure[8]
        ozone_noon = dust_figure[9]
        dust += ', 내일 오전 오존 상태는 ' + ozone_morn + ', 오후 상태는 ' + ozone_noon + '입니다'

        if '나쁨' in dust:
            dust += ' 공기 상태가 나쁘니 마스크를 착용하세요'
    return dust


def after_tomorrow_dust(location):
    if len(location.split()) == 1 and location in metropolitans:
        dust = metropolitan('내일', location)
    else:
        enc_location = urllib.parse.quote(location + '+ 내일 미세먼지')
        url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location

        req = Request(url)
        page = urlopen(req)
        html = page.read()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        dust_figure = soup.find_all('tbody')[2].text.split()
        dust_figure.remove('미세먼지')
        dust_figure.remove('초미세먼지')
        dust_figure.remove('오존')
        dust_figure.remove('자외선')
        dust_figure.remove('황사')

        dust = '모레 ' + location + '의 미세먼지 정보를 알려드릴게요 '
        dust_morn = dust_figure[2]
        dust_noon = dust_figure[3]
        dust += ', 모레 오전 미세먼지 상태는 ' + dust_morn + ', 오후 상태는 ' + dust_noon
        supdust_morn = dust_figure[6]
        supdust_noon = dust_figure[7]
        dust += ', 모레 오전 초미세먼지 상태는 ' + supdust_morn + ', 오후 상태는 ' + supdust_noon
        ozone_morn = dust_figure[10]
        ozone_noon = dust_figure[11]
        dust += ', 모레 오전 오존 상태는 ' + ozone_morn + ', 오후 상태는 ' + ozone_noon + '입니다'

        if '나쁨' in dust:
            dust += ' 공기 상태가 나쁘니 마스크를 착용하세요'

    return dust.replace('-', '아직 알수 없음')
