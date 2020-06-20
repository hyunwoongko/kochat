import json
import os
import re

import pandas as pd
import torch
from konlpy.tag import Okt
from pandas import DataFrame
from requests import Session

from backend.decorators import data


@data
class Preprocessor:
    """
    Class Preprocessor : @data (DATA configuration)

    메모리에 저장된 RAW 데이터파일을 불러와서 하나의 혼합파일로 만들고,
    Dataset 클래스가 학슴용 데이터를 만들 때 사용하는 여러가지 전처리
    기능이 구현된 클래스입니다.
    """

    def __init__(self):
        """
        텍스트 토큰화를 위해
        Konlpy의 Okt 태거 (구 Twitter 태거)를 사용합니다.
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
            intent_kinds = file_name.split('.')[0]
            intent_file = [(question, intent_kinds) for question in intent_question]
            intent_integrated_list += intent_file

        intent_df = DataFrame(intent_integrated_list, columns=['question', 'intent'])
        intent_df.to_csv(self.intent_data_dir, index=False)
        label, index = intent_df['intent'], -1

        for l in label:
            if l not in label_dict:
                index += 1
            label_dict[l] = index

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
            self.__check_label_number(entity_file)
            # TOKEN : ENTITY가 1 : 1이 되게끔 보장하는 함수
            # [안녕, 부산, 맛집, 추천해줘] 4 : 3 [O, LOCATION, O] → 에러 발생

            entity_file = [(q.strip().split(), l.strip().split())
                           for q, l in zip(entity_file['question'], entity_file['label'])]

            entity_integrated_list += entity_file
            for question, label in entity_file:
                for entity in label:
                    entity_non_duplicated_set.add(entity)

        self.__check_label_kinds(entity_non_duplicated_set)
        # 사용자가 정의하지 않은 라벨이 나오지 않게 보장하는 함수
        # 사용자 정의 태그 (LOCATION, DATE 등..)와 기능 태그 (B, I, O 등)의 조합으로
        # 있어서는 안될 태그(오타 등)를 발견시 에러를 발생시킵니다.

        entity_integrated_list = DataFrame([[' '.join(q), ' '.join(l)]
                                            for q, l in entity_integrated_list])

        entity_integrated_list.to_csv(path_or_buf=self.entity_data_dir,
                                      index=False,
                                      header=['question', 'entity'])

        entity_non_duplicated_set = sorted(list(entity_non_duplicated_set))
        for i, entity in enumerate(entity_non_duplicated_set):
            label_dict[entity] = i

        return label_dict

    def pad_sequencing(self, sequence: torch.Tensor) -> tuple:
        """
        패드 시퀀싱 함수입니다.
        max_len보다 길이가 길면 자르고, 짧으면 패딩(영벡터)를 추가합니다.
        엔티티 학습시에 CRF나 Masking 등을 이용하기 위해 각 문장의 길이가 필요합니다.
        패드 시퀀싱 단계에서는 어차피 길이를 세기 때문에 여기에서 길이를 반환합니다.

        :param sequence: 패드 시퀀싱할 문장입니다. (tensor로 이미 임베딩 된 문장)
        :return:
        """

        length = sequence.size()[0]
        if length > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            pad = torch.zeros(self.max_len, self.vector_size)
            for i in range(length):
                pad[i] = sequence[i]
            sequence = pad

        return sequence, length

    def label_sequencing(self, entity_label, entity_dict):
        length = entity_label.size()[0]

        if length > self.max_len:
            entity_label = entity_label[:self.max_len]
        else:
            while entity_label.size()[0] != self.max_len:
                outside_tag = entity_dict[self.NER_outside]
                outside_tag = torch.tensor(outside_tag).unsqueeze(0)
                entity_label = torch.cat([entity_label, outside_tag], dim=0)

        return entity_label.unsqueeze(0)

    def tokenize(self, sentence, train=False):
        if train:  # 학습데이터는 모두 맞춤법이 맞다고 가정
            return sentence.split()

        else:
            sentence = self.__naver_fix(sentence)
            sentence = self.__okt.pos(sentence)
            out = [word for word, pos in sentence
                   if pos not in ['Josa', 'Punctuation']]

            return self.__naver_fix(' '.join(out)).split()

    def __naver_fix(self, text):
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

    def __check_label_kinds(self, label_set: set):
        # Config에서 지정한 라벨들의 조합 + Outside 라벨만 가질 수 있음
        label = [tag + '-' + cate
                 for cate in self.NER_categories
                 for tag in self.NER_tagging] + [self.NER_outside]

        for entity in list(label_set):
            if entity not in label:
                raise Exception("THERE ARE LABEL ERROR : {}".format(entity))

    def __check_label_number(self, file):
        number_of_error = 0
        for i, (question, label) in enumerate(zip(file['question'].tolist(),
                                                  file['label'].tolist())):

            question = str(question).split(' ')
            entity = str(label).split(' ')

            if len(question) != len(entity):
                print(i - 2, question, entity)
                number_of_error += 1

        if number_of_error != 0:
            raise Exception("THERE ARE {} ERRORS!\n".format(number_of_error))

        return number_of_error
