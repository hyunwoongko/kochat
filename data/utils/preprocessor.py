"""
@auther Hyunwoong
@since {6/23/2020}
@see : https://github.com/gusdnd852
"""
import json
import re

import torch
from torch import Tensor
from konlpy.tag import Okt
from requests import Session

from _backend.decorators import data


@data
class Preprocessor:

    def __init__(self):
        """
        데이터를 전처리하는 여러가지 기능읃 가진 클래스입니다.
        패드시퀀싱, 토큰화, 맞춤법 교정등의 기능을 제공합니다.
        """

        self.okt = Okt()

    def pad_sequencing(self, sequence: Tensor) -> tuple:
        """
        패드 시퀀싱 함수입니다.
        max_len보다 길이가 길면 자르고, 짧으면 뒤에 패딩(영벡터)를 추가합니다.
        엔티티 학습시에 CRF나 Masking 등을 이용하기 위해 각 문장의 길이가 필요합니다.
        패드 시퀀싱 단계에서는 어차피 길이를 세기 때문에 길이를 함께 반환합니다.

        :param sequence: 패드 시퀀싱할 문장입니다. (tensor로 이미 임베딩 된 문장)
        :return: 패드시퀀싱된 문장과 시퀀싱 전의 문장의 원래 길이
        """

        length = sequence.size()[0]
        if length > self.max_len:
            sequence = sequence[:self.max_len]
            length = self.max_len  # 마스킹시에 길이가 max_len 넘어가면 안됨
            # 문장이 max_len보다 길면 뒷부분을 자릅니다.

        else:
            pad = torch.zeros(self.max_len, self.vector_size)
            for i in range(length):
                pad[i] = sequence[i]
            sequence = pad
            # 문장이 max_len보다 짧으면 길이가 max_len인 0벡터를 만들고
            # 데이터가 있던 인덱스에는 원래 데이터를 복사합니다

        return sequence, length

    def label_sequencing(self, entity_label: Tensor, entity_dict: dict) -> Tensor:
        """
        엔티티 라벨의 경우에는 라벨도 각각 길이가 다르게 됩니다.
        e.g. [O, DATE, O](size=3),  [DATE, O, O, O](size=4)
        길이가 다른 벡터들을 텐서의 형태로 만들려면 이들의 길이도 같아야합니다.

        :param entity_label: 한 문장의 엔티티 라벨 (1차원)
        :param entity_dict: 딕셔너리를 이용해 빈부분에 outside 태그를 넣습니다.
        :return: 패드시퀀싱 된 엔티티 라벨
        """

        length = entity_label.size()[0]

        if length > self.max_len:
            entity_label = entity_label[:self.max_len]
            # 길이가 max_len보다 길면 뒷부분을 자릅니다.

        else:
            pad = torch.ones(self.max_len, dtype=torch.int64)
            outside_tag = entity_dict[self.NER_outside]
            pad = pad * outside_tag  # 'O' 태그가 맵핑된 숫자
            # [1, 1, ..., 1] * 'O' => ['O', 'O', ... , 'O']

            for i in range(length):
                pad[i] = entity_label[i]
            entity_label = pad
            # 문장이 max_len보다 짧으면 길이가 max_len인 'O'벡터를 만들고
            # 데이터가 있던 인덱스에는 원래 데이터를 복사합니다

        return entity_label.unsqueeze(0)

    def tokenize(self, sentence: str, train: bool = False, naver_fix: bool = True) -> list:
        """
        문장의 맞춤법을 교정하고 토큰화 합니다.
        유저의 입력문장의 경우에만 맞춤법 교정을 진행하고,
        학습/테스트 데이터는 띄어쓰기 기준으로 자릅니다.

        :param sentence: 토큰화할 문장
        :param train: 학습모드 여부 (True이면 맞춤법 교정 X)
        :param naver_fix: 네이버 맞춤법 검사기 사용 여부
        :return: 토큰화된 문장
        """

        if train:  # 학습데이터는 모두 맞춤법이 맞다고 가정 (속도 향상위해)
            return sentence.split()

        else:  # 사용자 데이터는 전처리를 과정을 거침 (fix → tok → fix)
            if naver_fix:
                sentence = self.__naver_fix(sentence)

            sentence = self.okt.pos(sentence)

            # 조사와 구두점은 잘라냅니다.
            out = [word for word, pos in sentence
                   if pos not in ['Josa', 'Punctuation']]

            if naver_fix:
                return self.__naver_fix(' '.join(out)).split()

            return out

    def __naver_fix(self, text: str) -> str:
        """
        ajax 크롤링을 이용하여 네이버 맞춤법 검사기 API를 사용합니다.

        :param text: 맞춤법을 수정할 문장
        :return: 맞춤법이 수정된 문장
        """

        if len(text) > 500:
            raise Exception('500글자 이상 넘을 수 없음!')

        sess = Session()

        # ajax 크롤링을 이용합니다 (네이버 맞춤법 검사기)
        data = sess.get(
            url='https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn',
            params={
                '_callback':
                    'window.__jindo2_callback._spellingCheck_0',
                'q': text},
            headers={
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
                'referer': 'https://search.naver.com/'
            }
        )

        data = json.loads(data.text[42:-2])  # json 파싱
        html = data['message']['result']['html']  # 원하는부분 가져오기
        out = re.sub(re.compile('<.*?>'), '', html)  # tag 잘라내기
        return out
