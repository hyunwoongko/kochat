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
class Preprocessor():

    def __init__(self):
        super().__init__()
        self.__entity_set = set()
        self.__okt = Okt()

    def generate_intent(self):
        files = os.listdir(self.raw_data_dir)
        intent_files = []

        for file_name in files:
            intent = file_name.split('.')[0]
            intent_file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
            question = intent_file['question'].values.tolist()
            intent_file = [(data, intent) for data in question]
            intent_files += intent_file

        DataFrame(intent_files).to_csv(path_or_buf=self.intent_data_dir,
                                       index=False,
                                       header=['question', 'intent'])

    def generate_entity(self):
        files = os.listdir(self.raw_data_dir)
        entity_files = []

        for file_name in files:
            entity_file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
            self.__check_label_number(entity_file)
            # QUESTION 수 : ENTITY 수가 1 : 1이 되게끔 보장하는 함수

            question = entity_file['question'].values.tolist()
            entity = entity_file['label'].values.tolist()
            entity_file = [(data[0].strip().split(),
                            data[1].strip().split())
                           for data in zip(question, entity)]

            entity_files += entity_file
            for entity in entity_file:
                for e in entity[1]:
                    self.__entity_set.add(e)
                    # Entity Set 만들기
                    # (생성자 속성으로 있는 entity set을 이 때 만듬)

        # 사용자가 정의하지 않은 라벨이 나오지 않게 보장하는 함수
        self.__check_label_kinds(label_set=self.__entity_set)
        entity_files = DataFrame([[' '.join(data[0]), ' '.join(data[1])] for data in entity_files])
        entity_files.to_csv(path_or_buf=self.entity_data_dir,
                            index=False,
                            header=['question', 'entity'])
        return self.__entity_set

    def make_dicts(self):
        intent_dict, entity_dict = {}, {}
        dataset = pd.read_csv(self.intent_data_dir)
        label, index = dataset['intent'], -1

        # label to int
        for lb in label:
            if lb not in intent_dict:
                index += 1
            intent_dict[lb] = index

        label_set = self.__entity_set
        label_set = list(label_set)
        label_set = sorted(label_set)
        # 항상 동일한 결과를 보여주려면 정렬해놔야 함

        for i, entity in enumerate(label_set):
            entity_dict[entity] = i

        return intent_dict, entity_dict

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
        for i, data in enumerate(zip(file['question'].tolist(),
                                     file['label'].tolist())):

            q = str(data[0]).split(' ')
            e = str(data[1]).split(' ')

            if len(q) != len(e):
                # Question과 Entity의 수가 다를때
                print(i - 2, q, e)
                # 화면에 보여주고 에러 개수 1개 늘림
                number_of_error += 1

        if number_of_error != 0:
            raise Exception("THERE ARE {} ERRORS!\n".format(number_of_error))

        return number_of_error
