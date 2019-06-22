import random
import re
import urllib
from urllib.request import urlopen, Request

import bs4


def recommend_restaurant(location):
    enc_location = urllib.parse.quote(location + '맛집')
    url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location

    rand = random.randint(0, 5)
    req = Request(url)
    page = urlopen(req)
    html = page.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    list_name = soup.find_all('a', class_='name')
    list_info = soup.find_all('div', class_='txt ellp')

    name = list_name[rand].text.split()
    del name[0]
    name = ' '.join(name)
    info = list_info[rand].text

    specific_url = list_name[rand].get('href')
    req = Request(specific_url)
    page = urlopen(req)
    html = page.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    document = soup.find_all('div', {'class': 'txt'})

    tel = ''
    if document[0] is not None:
        tel = document[0].text

    addr = ''
    if document[1].find('span', {'class': 'addr'}) is not None:
        addr = document[1].find('span', {'class': 'addr'}).text

    time = ''
    if document[2].find('span', {'class': 'time'}) is not None:
        time = document[2].find('span', {'class': 'time'}).text
        time = re.sub("-", " 에서 ", time)

    if document[3].find_all('em', {'class': 'price'}) is not None:
        price_list = document[3].find_all('em', {'class': 'price'})
        menu_list = document[3].find_all('span', {'class': 'name'})

        menu_size = len(price_list)
        menu = []
        menu_dict = {}
        for i in range(menu_size):
            for p in price_list, menu_list:
                menu.append(p[i].text)
        for i in range(len(menu)):
            if i % 2 == 0:
                menu_dict[menu[i + 1]] = menu[i]

    link_path = soup.find('ul', {'class': 'list_relation_link'})
    if link_path is not None:

        link = link_path.find_all('li', {'class': 'list_item'})
        siksin = ''
        for i in link:
            link_spceific = i.find('a').get('href')
            if 'siksinhot' in link_spceific:
                siksin = link_spceific
        if siksin != '':
            req = Request(siksin)
            page = urlopen(req)
            html = page.read()
            soup = bs4.BeautifulSoup(html, 'html.parser')
            siksin_doc = soup.find('div', {'itemprop': 'articleBody'}).text.split()

            counter = False
            response_list = []
            for word in siksin_doc:
                word = re.sub('하다', '합니다', word)
                word = re.sub('한다', '합니다', word)
                word = re.sub('했다', '했어요', word)
                word = re.sub('했었다', '했었어요', word)
                word = re.sub('이다', '입니다', word)
                word = re.sub('있다', '있어요', word)
                word = re.sub('있었다', '있었어요', word)

                if '전화번호' in word:
                    response_list.append(word.split(sep='전화번호', maxsplit=1)[0])
                    counter = False
                if counter:
                    response_list.append(word)
                if '매장소개' in word:
                    response_list.append(word.split(sep='매장소개', maxsplit=1)[1])
                    counter = True

            description = ' '.join(response_list)
        else:
            description = ''
    else:
        description = ''

    msg = info + ' , ' + name + ' 에 가보는 건 어떨까요? '

    if description != ' ':
        msg += description

    if tel != '':
        msg += '  운영시간은 ' + time + ','

    if addr != '':
        msg += ' 주소는 ' + addr + ','

    if tel != '':
        msg += ' 전화번호는 ' + tel + ','

    msg += '입니다.'

    return msg
