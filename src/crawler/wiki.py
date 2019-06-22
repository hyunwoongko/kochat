# -*- coding: utf-8 -*-
import urllib
from urllib.request import urlopen, Request

import bs4


def wiki(question):
    parsed_question = urllib.parse.quote(question)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        'referer': 'http://google.com'}
    url = 'https://www.google.com/search?q=' + parsed_question

    req = Request(url, headers=headers)
    page = urlopen(req)
    html = page.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    response_list = []

    try:
        data = soup.find("div", class_='SALvLe farUxc mJ2Mod').text
        exp_list = data.split()

        counter = False
        terminator = '설명'

        if '설명' not in question:
            for word in exp_list:
                if '위키백과' in word:
                    counter = False
                if counter:
                    response_list.append(word)
                if '설명' in word:
                    response_list.append(word.split(sep=terminator, maxsplit=1)[1])
                    counter = True
            response_text = ' '.join(response_list)
        else:
            response_text = '설명은 어떤 사실이나 정보, 지식 등을 정확하고 알기 쉽게 전달하여 이해시키는 것을 목적으로 하는 진술 방법이다. ' \
                            '철학에서는 "있는 사건의 근거를 법률에서 연역적으로 찾는 것"을 말한다.'
    except:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            'referer': 'http://daum.net'}
        url = 'https://alldic.daum.net/search.do?q=' + parsed_question + '&dic=kor'

        req = Request(url, headers=headers)
        page = urlopen(req)
        html = page.read()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        try:
            result = (soup
                      .find('div', class_='cleanword_type kokk_type')
                      .find('ul', class_='list_search')
                      .find_all('li'))
            result_set = [i.text.split(sep='.', maxsplit=1)[1] for i in result]
            response_text = ' 또는 '.join(result_set)
        except:
            response_text = '죄송해요 잘 못 알아들었어요. 조금 더 자세히 말해주세요.'

    if len(response_list) == 0:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            'referer': 'http://daum.net'}
        url = 'https://alldic.daum.net/search.do?q=' + parsed_question + '&dic=kor'

        req = Request(url, headers=headers)
        page = urlopen(req)
        html = page.read()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        try:
            result = (soup
                      .find('div', class_='cleanword_type kokk_type')
                      .find('ul', class_='list_search')
                      .find_all('li'))
            result_set = [i.text.split(sep='.', maxsplit=1)[1] for i in result]
            response_text = ' 또는 '.join(result_set)
        except:
            response_text = '죄송해요 잘 못 알아들었어요. 조금 더 자세히 말해주세요.'
    return response_text
