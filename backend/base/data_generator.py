import os

import pandas as pd
from pandas import DataFrame

from backend.base.data_manager import DataManager


class DataGenerator(DataManager):

    def __init__(self):
        super().__init__()
        self.entity_set = set()

    def _generate_intent(self):
        """
        FILE INTENT :
        QUESTION, INTENT1
        QUESTION, INTENT1
        QUESTION, INTENT2
        QUESTION, INTENT2
        QUESTION, INTENT3
        QUESTION, ...

        RAW 데이터를 읽어들여서 위와 같은 형식의 Intent 데이터셋을 만듭니다.
        이 때, INTENT의 이름은 파일명을 따라갑니다.
        """

        files = os.listdir(self.raw_data_dir)
        intent_files = []

        for file_name in files:
            intent = file_name.split('.')[0]
            intent_file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
            question = intent_file['question'].values.tolist()
            intent_file = [(data, intent) for data in question]
            intent_files += intent_file

        DataFrame(intent_files).to_csv(path_or_buf=self.intent_data_file,
                                       index=False,
                                       header=['question', 'intent'])

    def _generate_entity(self):
        """
        FILE INTENT :
        [WORD1 WORD2 WORD3 ...], [ENTITY1, ENTITY2, ENTITY3 ...]
        [WORD1 WORD2 WORD3 ...], [ENTITY1, ENTITY2, ENTITY3 ...]
        [WORD1 WORD2 WORD3 ...], [ENTITY1, ENTITY2, ENTITY3 ...]

        데이터를 읽어들여서 위와 같은 형식의 Entity 데이터셋을 만듭니다.
        Entity는 데이터 작성시 실수할 확률이 매우 크므로 몇가지 체크 코드가 들어갑니다.
        """

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
        entity_files.to_csv(path_or_buf=self.entity_data_file,
                            index=False,
                            header=['question', 'entity'])

    def _check_label_kinds(self, label_set: set):
        """
        Config에 지정한 라벨 이외의 라벨(오타 등에 의한 실수)가 있으면
        어떻게 틀렸는지 화면에 보여주고, 예외를 발생시킵니다.
        데이터 제작시 사소한 실수가 발생하면 Ctrl + F로 찾아서 고칠 수 있습니다.

        :param label_set: 현재 읽은 데이터에서 뽑아낸 entity들의 Set입니다 (중복X)
        :param categories: 사용자가 Config 파일에 지정한 카테고리들의 목록입니다.
        :param tags: 사용자가 Config 파일에 지정한 태그 (Begin, End 등)의 목록입니다.
        :return:
        """

        # Config에서 지정한 라벨들의 조합 + non_tag만 가질 수 있음
        label = [tag + '-' + cate
                 for cate in self.NER_categories
                 for tag in self.NER_tagging] + [self.NER_outside]

        for entity in list(label_set):
            if entity not in label:
                raise Exception("THERE ARE LABEL ERROR : {}".format(entity))

    @staticmethod
    def _check_label_number(file):
        """
        QUESTION, ENTITY Pair에서 QUESTION 수와 ENTITY 수를 비교합니다.
        수가 안 맞으면 화면에 어디에서 틀렸는지 보여주고 예외를 발생시킵니다.

        :param file: pandas로 읽어온 dataframe
        :return: 오류의 개수를 리턴합니다.
        """
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
