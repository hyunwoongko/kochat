import json
import re
import torch
import config

from konlpy.tag import Okt
from requests import Session
from base.base_component import BaseComponent


class DataManager(BaseComponent):

    def __init__(self):
        super().__init__()
        self.okt = Okt()
        for key, val in config.DATA.items():
            setattr(self, key, val)

    def pad_sequencing(self, sequence):
        size = sequence.size()[0]
        if size > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            pad = torch.zeros(self.max_len, self.vector_size)
            for i in range(size):
                pad[i] = sequence[i]
            sequence = pad

        return sequence

    def tokenize(self, sentence, train=False):
        if train:  # 학습데이터는 모두 맞춤법이 맞다고 가정
            return sentence.split()

        else:
            sentence = self._naver_fix(sentence)
            sentence = self.okt.pos(sentence)
            out = [word for word, pos in sentence
                   if pos not in ['Josa', 'Punctuation']]

            return self._naver_fix(' '.join(out)).split()

    @staticmethod
    def _naver_fix(text):
        if len(text) > 500:
            raise Exception('500글자 이상 넘을 수 없음!')

        sess = Session()
        data = sess.get(url='https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn',
                        params={
                            '_callback':
                                'window.__jindo2_callback._spellingCheck_0',
                            'q': text},
                        headers={
                            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
                            'referer': 'https://search.naver.com/'
                        })

        data = json.loads(data.text[42:-2])
        html = data['message']['result']['html']
        out = re.sub(re.compile('<.*?>'), '', html)
        return out
