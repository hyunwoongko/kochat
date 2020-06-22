import json
import os
import re

import pandas as pd
import torch
from konlpy.tag import Okt
from pandas import DataFrame
from requests import Session

from _backend.decorators import data


@data
class Preprocessor:

    def __init__(self):
        """
        메모리에 저장된 RAW 데이터파일을 불러와서 하나의 혼합파일로 만들고,
        Dataset 클래스가 학슴용 데이터를 만들 때 사용하는 여러가지 전처리
        기능이 구현된 클래스입니다.
        """

        self.__okt = Okt()

    def generate_intent(self) -> dict:
        """
        여러 파일을 모아서 하나의 인텐트 데이터 파일을 생성합니다.
        인텐트는 파일 단위로 구분되며, 파일명이 인텐트가 됩니다.
        이런 방식으로 파일을 구성하면 인텐트/엔티티 데이터를 따로 만들지 않아도 됩니다.

        :return: 인텐트 라벨들이 순차적으로 숫자로 맵핑된 Dictionary입니다.
        """

        files = os.listdir(self.raw_data_dir)
        intent_integrated_list, label_dict = [], {}
        # 개별 파일이 담긴 뒤 합쳐질 리스트와 라벨 딕셔너리

        for file_name in files:
            intent_file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
            intent_question = intent_file['question'].values.tolist()
            intent_kinds = file_name.split('.')[0]  # [ABC.csv] 에서 .보다 앞부분(ABC)을 인텐트 종류로 구분
            intent_file = [(question, intent_kinds) for question in intent_question]  # question과 label 세트
            intent_integrated_list += intent_file

        intent_df = DataFrame(intent_integrated_list, columns=['question', 'intent'])
        intent_df.to_csv(self.intent_data_dir, index=False)
        label, index = intent_df['intent'], -1

        for l in label:
            if l not in label_dict:
                index += 1
            label_dict[l] = index
            # 라벨 딕셔너리 만듬

        return label_dict

    def generate_entity(self) -> dict:
        """
        여러 파일을 모아서 하나의 엔티티 데이터 파일을 생성합니다.
        엔티티는 라벨링이 직접 되어있기 때문에 데이터의 라벨링을 따릅니다.
        이런 방식으로 파일을 구성하면 인텐트/엔티티 데이터를 따로 만들지 않아도 됩니다.

        :return: 엔티티 라벨(태그)들이 순차적으로 숫자로 맵핑된 Dictionary입니다.
        """

        files = os.listdir(self.raw_data_dir)
        entity_integrated_list, label_dict = [], {}
        entity_non_duplicated_set = set()
        # 개별 파일이 담긴 뒤 합쳐질 리스트와 라벨 딕셔너리

        for file_name in files:
            entity_file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
            self.__check_num_of_label(entity_file)
            # TOKEN : ENTITY가 1 : 1이 되게끔 보장하는 함수

            entity_file = [(q.strip().split(), l.strip().split())  # 공백 없애고 단어단위로 자름
                           for q, l in zip(entity_file['question'], entity_file['label'])]

            entity_integrated_list += entity_file
            for question, label in entity_file:
                for entity in label:
                    entity_non_duplicated_set.add(entity)

        self.__check_label_kinds(entity_non_duplicated_set)
        # 사용자가 정의하지 않은 라벨이 나오지 않게 보장하는 함수

        entity_integrated_list = \
            DataFrame([[' '.join(q), ' '.join(l)] for q, l in entity_integrated_list])
        # [오늘, 날씨, 알려줘]와 [S-DATE, O, O]처럼 리스트로 된 데이터를 join하여
        # "오늘 날씨 알려줘"와 "S-DATE O O"와 같이 만들어서 저장합니다.

        entity_integrated_list.to_csv(path_or_buf=self.entity_data_dir,
                                      index=False,
                                      header=['question', 'entity'])

        entity_non_duplicated_set = sorted(list(entity_non_duplicated_set))
        # 매번 같은 인덱스를 갖게 하려면 정렬해야합니다.

        for i, entity in enumerate(entity_non_duplicated_set):
            label_dict[entity] = i  # 엔티티 딕셔너리 만듬

        return label_dict

    def pad_sequencing(self, sequence: torch.Tensor) -> tuple:
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
            length = 8  # 마스킹시에 길이가 max_len 넘어가면 안됨
            # 문장이 max_len보다 길면 뒷부분을 자릅니다.

        else:
            pad = torch.zeros(self.max_len, self.vector_size)
            for i in range(length):
                pad[i] = sequence[i]
            sequence = pad
            # 문장이 max_len보다 짧으면 길이가 max_len인 0벡터를 만들고
            # 데이터가 있던 인덱스에는 원래 데이터를 복사합니다

        return sequence, length

    def label_sequencing(self, entity_label, entity_dict):
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

    def tokenize(self, sentence, train=False):
        """
        문장의 맞춤법을 교정하고 토큰화 합니다.
        유저의 입력문장의 경우에만 맞춤법 교정을 진행하고,
        학습/테스트 데이터는 띄어쓰기 기준으로 자릅니다.

        :param sentence: 토큰화할 문장
        :param train: 학습모드 여부 (True이면 맞춤법 교정 X)
        :return: 토큰화된 문장
        """

        if train:  # 학습데이터는 모두 맞춤법이 맞다고 가정 (속도 향상위해)
            return sentence.split()

        else:  # 사용자 데이터는 전처리를 과정을 거침 (fix → tok → fix)
            sentence = self.__naver_fix(sentence)
            sentence = self.__okt.pos(sentence)

            # 조사와 구두점은 잘라냅니다.
            out = [word for word, pos in sentence
                   if pos not in ['Josa', 'Punctuation']]

            return self.__naver_fix(' '.join(out)).split()

    def __naver_fix(self, text):
        """
        ajax 크롤링을 이용하여 네이버 맞춤법 검사기 API를 사용합니다.

        :param text: 맞춤법을 수정할 문장
        :return: 맞춤법이 수정된 문장
        """
        if len(text) > 500:
            raise Exception('500글자 이상 넘을 수 없음!')

        sess = Session()
        # ajax 크롤링을 이용합니다 (네이버 맞춤법 검사기)
        data = sess.get(url='https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn',
                        params={
                            '_callback':
                                'window.__jindo2_callback._spellingCheck_0',
                            'q': text},
                        headers={
                            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
                            'referer': 'https://search.naver.com/'
                        })

        data = json.loads(data.text[42:-2])  # json 파싱
        html = data['message']['result']['html']  # 원하는부분 가져오기
        out = re.sub(re.compile('<.*?>'), '', html)  # tag 잘라내기
        return out

    def __check_label_kinds(self, label_set: set):
        """
        사용자가 정의한 태그 이외의 태그를 가지면 오류를 발생시킵니다.
        
        :param label_set: 라벨세트
        :exception: 만약 사용자가 정의한 태그 이외의 태그가 있는 경우
        """

        # Config에서 지정한 라벨들의 조합 + Outside 라벨만 가질 수 있음
        label = [tag + '-' + cate
                 for cate in self.NER_categories
                 for tag in self.NER_tagging] + [self.NER_outside]

        for entity in list(label_set):
            if entity not in label:
                raise Exception("THERE ARE LABEL ERROR : {}".format(entity))
                # 에러가 발생한 부분의 엔티티(라벨)를 보여줌

    def __check_num_of_label(self, data_df):
        """
        데이터(question)부분과 레벨(entity)부분의 갯수가 안맞으면 에러를 발생시킵니다.
        e.g. [오늘, 전주, 날씨, 어떠니](size=4) , [DATE, LOCATION, O](size=3) → 에러발생

        :param file: 에러를 체크할 데이터 프레임입니다.
        :exception: 만약 위와 같이 question과 entity의 수가 다르면 에러를 발생시킵니다.
        :return: 해당 데이터프레임 내의 에러 개수를 리턴합니다.
        """

        number_of_error = 0
        for i, (question, label) in enumerate(zip(data_df['question'].tolist(),
                                                  data_df['label'].tolist())):

            question = str(question).split(' ')
            entity = str(label).split(' ')

            if len(question) != len(entity):
                print(i - 2, question, entity)
                number_of_error += 1

        if number_of_error != 0:
            raise Exception("THERE ARE {} ERRORS!\n".format(number_of_error))

        return number_of_error
