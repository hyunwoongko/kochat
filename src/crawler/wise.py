import random
import urllib.request

import bs4


def get_wise():
    wise_list = ['사랑', '인생', '공부', '성공', '친구', '독서', '이별', '시간', '노력', '희망', '도전', '자신감']
    parsed_wise = urllib.parse.quote(random.choice(wise_list) + ' 명언')
    fullUrl = 'https://search.naver.com/search.naver?where=nexearch&sm=tab_etc&mra=blMy&query=' + parsed_wise
    req = urllib.request.Request(fullUrl, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read()
    bsObj = bs4.BeautifulSoup(html, "html.parser")
    wsStudy = bsObj.find("p", {"class": "lngkr"}).text
    return wsStudy
