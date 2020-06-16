import os
from abc import ABCMeta
import pandas as pd
from konlpy.tag import Okt
from pandas import DataFrame
from backend.decorators import data


@data
class Dataset(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.okt = Okt()
        self.entity_set = set()

    def _generate_intent(self):
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

    def _generate_entity(self):
        files = os.listdir(self.raw_data_dir)
        entity_files = []

        for file_name in files:
            entity_file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
            self._check_label_number(entity_file)
            # QUESTION 수 : ENTITY 수가 1 : 1이 되게끔 보장하는 함수

            question = entity_file['question'].values.tolist()
            entity = entity_file['label'].values.tolist()
            entity_file = [(data[0].strip().split(),
                            data[1].strip().split())
                           for data in zip(question, entity)]

            entity_files += entity_file
            for entity in entity_file:
                for e in entity[1]:
                    self.entity_set.add(e)
                    # Entity Set 만들기
                    # (생성자 속성으로 있는 entity set을 이 때 만듬)

        # 사용자가 정의하지 않은 라벨이 나오지 않게 보장하는 함수
        self._check_label_kinds(label_set=self.entity_set)
        entity_files = DataFrame([[' '.join(data[0]), ' '.join(data[1])] for data in entity_files])
        entity_files.to_csv(path_or_buf=self.entity_data_dir,
                            index=False,
                            header=['question', 'entity'])

    def _check_label_kinds(self, label_set: set):
        # Config에서 지정한 라벨들의 조합 + non_tag만 가질 수 있음
        label = [tag + '-' + cate
                 for cate in self.NER_categories
                 for tag in self.NER_tagging] + [self.NER_outside]

        for entity in list(label_set):
            if entity not in label:
                raise Exception("THERE ARE LABEL ERROR : {}".format(entity))

    def _check_label_number(self, file):
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
