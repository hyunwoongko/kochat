"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict

import requests
from konlpy.tag import Okt

from collections import namedtuple

base_url = 'https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn'
_checked = namedtuple('Checked', ['result', 'original', 'checked', 'errors', 'words', 'time'])


class CheckResult:
    PASSED = 0
    WRONG_SPELLING = 1
    WRONG_SPACING = 2
    AMBIGUOUS = 3


class Checked(_checked):
    def __new__(cls, result=False, original='', checked='', errors=0, words=[], time=0.0):
        return super(Checked, cls).__new__(
            cls, result, original, checked, errors, words, time)

    def as_dict(self):
        d = {
            'result': self.result,
            'original': self.original,
            'checked': self.checked,
            'errors': self.errors,
            'words': self.words,
            'time': self.time,
        }
        return d


class Tokenizer:
    """
    Naver Spellchecker + Okt Tokenizer
    """
    okt = Okt()
    _agent = requests.Session()
    PY3 = sys.version_info[0] == 3

    josa = [
        '이구나', '이네', '이야',
        '은', '는', '이', '가', '을', '를',
        '로', '으로', '이야', '야', '냐', '니'
    ]

    def _remove_tags(self, text):
        text = u'<content>{}</content>'.format(text).replace('<br>', '')
        if not self.PY3:
            text = text.encode('utf-8')

        result = ''.join(ET.fromstring(text).itertext())

        return result

    def check(self, text):

        if isinstance(text, list):
            result = []
            for item in text:
                checked = self.check(item)
                result.append(checked)
            return result

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
        r = self._agent.get(base_url, params=payload, headers=headers)
        passed_time = time.time() - start_time

        r = r.text[42:-2]
        data = json.loads(r)
        html = data['message']['result']['html']
        result = {
            'result': True,
            'original': text,
            'checked': self._remove_tags(html),
            'errors': data['message']['result']['errata_count'],
            'time': passed_time,
            'words': OrderedDict(),
        }

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
        return result.checked

    def tokenize(self, sentence, train=False):
        if train:  # 모든 데이터의 맞춤법이 맞다고 가정
            return sentence.split()

        else:  # 유저 입력시 토크나이징
            sentence = self.check(sentence)
            pos_list = self.okt.pos(sentence)
            result = [word for word, pos in pos_list if word not in self.josa and pos != 'Punctuation']
            return self.check(''.join(result)).split()
