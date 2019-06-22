# -*- coding: utf-8 -*-
"""
Python용 한글 맞춤법 검사 모듈
"""

import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict

import pandas as pd
import requests

from .constants import CheckResult
from .constants import base_url
from .response import Checked

_agent = requests.Session()
PY3 = sys.version_info[0] == 3
my_dict = pd.read_csv('util/hanspell/spell_dict.csv')
spell_dict = {}

for k, v in my_dict.values:
    spell_dict[k] = v


def _remove_tags(text):
    text = u'<content>{}</content>'.format(text).replace('<br>', '')
    if not PY3:
        text = text.encode('utf-8')

    result = ''.join(ET.fromstring(text).itertext())

    return result


def check(text):
    """
    매개변수로 입력받은 한글 문장의 맞춤법을 체크합니다.
    """
    if isinstance(text, list):
        result = []
        for item in text:
            checked = check(item)
            result.append(checked)
        return result

    # 최대 500자까지 가능.
    if len(text) > 500:
        return Checked(result=False)

    payload = {
        '_callback': 'window.__jindo2_callback._spellingCheck_0',
        'q': text
    }

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'referer': 'https://search.naver.com/'
    }

    start_time = time.time()
    r = _agent.get(base_url, params=payload, headers=headers)
    passed_time = time.time() - start_time

    r = r.text[42:-2]

    data = json.loads(r)
    html = data['message']['result']['html']
    result = {
        'result': True,
        'original': text,
        'checked': _remove_tags(html),
        'errors': data['message']['result']['errata_count'],
        'time': passed_time,
        'words': OrderedDict(),
    }

    # 띄어쓰기로 구분하기 위해 태그는 일단 보기 쉽게 바꿔둠.
    # ElementTree의 iter()를 써서 더 좋게 할 수 있는 방법이 있지만
    # 이 짧은 코드에 굳이 그렇게 할 필요성이 없으므로 일단 문자열을 치환하는 방법으로 작성.
    html = html.replace('<span class=\'re_green\'>', '<green>') \
        .replace('<span class=\'re_red\'>', '<red>') \
        .replace('<span class=\'re_purple\'>', '<purple>') \
        .replace('</span>', '<end>')
    items = html.split(' ')
    words = []
    tmp = ''
    for word in items:
        if tmp == '' and word[:1] == '<':
            pos = word.find('>') + 1
            tmp = word[:pos]
        elif tmp != '':
            word = u'{}{}'.format(tmp, word)

        if word[-5:] == '<end>':
            word = word.replace('<end>', '')
            tmp = ''

        words.append(word)

    for word in words:
        check_result = CheckResult.PASSED
        if word[:5] == '<red>':
            check_result = CheckResult.WRONG_SPELLING
            word = word.replace('<red>', '')
        elif word[:7] == '<green>':
            check_result = CheckResult.WRONG_SPACING
            word = word.replace('<green>', '')
        elif word[:8] == '<purple>':
            check_result = CheckResult.AMBIGUOUS
            word = word.replace('<purple>', '')

        result['words'][word] = check_result

    result = Checked(**result)

    return result


def exception(text):
    for key, val in spell_dict.items():
        if key in text:
            return text.replace(key, val)
    return text


def fix(text):
    if text is not None:
        result = check(text)
        result.as_dict()  # dict로 출력
        answer = exception(result[2])
        return answer
    return text
