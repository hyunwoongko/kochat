import os

import pandas as pd
from pandas import DataFrame, Series

from kochat.decorators import data


@data
class Organizer:

    def __init__(self):
        """
        메모리에 저장된 데이터파일을 불러와서 하나의 혼합파일로 만들고,
        학습시 필요한 라벨 딕셔너리를 생성하여 반환하는 클래스입니다.
        """

    def organize_intent(self) -> dict:
        """
        여러 파일을 모아서 하나의 인텐트 데이터 파일로 만듭니다.
        파일 저장 이후에 학습에 사용되는 딕셔너리를 만들고 반환합니다.

        :return: 인텐트 라벨들이 순차적으로 숫자로 맵핑된 Dictionary입니다.
        (e.g. {날씨: 1, 맛집: 2, 미세먼지:3, ...})
        """

        files = os.listdir(self.raw_data_dir)
        # 디렉토리에서 파일들의 리스트를 불러옴

        integrated_file = []
        # 개별 파일들의 데이터를 모두 저장할 통합 리스트

        for file_name in files:
            intent_file = self.__process_intent_file(file_name)
            integrated_file += intent_file
            # 개별 파일 단위 프로세싱 이후 하나의 파일로 통합

        intent_df = DataFrame(integrated_file, columns=['question', 'label'])
        intent_df.to_csv(self.intent_data_dir, index=False)
        intent_dict = self.__make_intent_dict(intent_df['label'])
        return intent_dict

    def organize_entity(self) -> dict:
        """
        여러 파일을 모아서 하나의 엔티티 데이터 파일로 만듭니다.
        파일 저장 이후에 학습에 사용되는 딕셔너리를 만들고 반환합니다.

        :return: 엔티티 라벨(태그)들이 순차적으로 숫자로 맵핑된 Dictionary입니다.
        (e.g. {B-DATE: 1, B-LOCATION: 2, B-RESTAURANT:3, ...})
        """

        files = os.listdir(self.raw_data_dir)
        # 디렉토리에서 파일들의 리스트를 불러옴

        integrated_file, label_set = [], set()
        # 개별 파일들의 데이터를 모두 저장할 통합 리스트

        for file_name in files:
            entity_file, labels = self.__process_entity_file(file_name)
            integrated_file += entity_file

            for label in labels:
                label_set.add(label)

        self.__check_label_kinds(label_set)  # 라벨 종류 체크
        entity_df = DataFrame([[' '.join(q), ' '.join(l)] for q, l in integrated_file])
        entity_df.to_csv(self.entity_data_dir, index=False, header=['question', 'label'])
        entity_dict = self.__make_entity_dict(label_set)
        return entity_dict

    def __process_intent_file(self, file_name: str) -> list:
        """
        개별 인텐트 파일 단위의 프로세싱입니다.
        파일명으로부터 인텐트를 뽑아내고 question문장과 인텐트 튜플리스트를 반환합니다.

        :param file_name: 개별 파일 이름 (XXX.csv)
        :return: 개별 샘플(question, label)들의 리스트 (2차원)
        """

        file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
        questions = file['question'].values.tolist()  # question만 뽑아서 리스트로
        intents = file_name.split('.')[0]  # [ABC.csv] 에서 '.'보다 앞부분(ABC)을 인텐트로 구분
        return [(question, intents) for question in questions]  # question과 label 세트 반환

    def __process_entity_file(self, file_name: str) -> tuple:
        """
        개별 엔티티 파일 단위의 프로세싱입니다.
        엔티티 데이터의 유효성을 검증한 뒤 (check_num_of_label, check_label_kinds)
        question과 labels들의 튜플리스트와 라벨 집합을 반환합니다.

        :param file_name: 개별 파일 이름 (XXX.csv)
        :return: 개별 샘플(question, label)들의 리스트 (2차원), 라벨 집합
        """

        entity_file = pd.read_csv(self.raw_data_dir + file_name, encoding='utf-8')
        self.__check_num_of_label(entity_file)  # question과 entity 개수 체크

        entity_file = [(question.strip().split(), labels.strip().split())
                       for question, labels
                       in zip(entity_file['question'], entity_file['label'])]

        labels = [label
                  for _, labels in entity_file
                  for label in labels]

        return entity_file, labels

    def __make_intent_dict(self, intents: Series) -> dict:
        """
        학습 등에 사용되는 인텐트 라벨 딕셔너리를 생성합니다.

        :param intents: 원본 데이터에서 인텐트 컬람만 뽑아낸 리스트
        :return: 인텐트 딕셔너리 (e.g. {날씨:1, 맛집:2, ...})
        """

        label_dict, index = {}, -1

        # 딕셔너리에 없는 인텐트는 새로 추가
        for intent in intents:
            if intent not in label_dict:
                index += 1

            label_dict[intent] = index

        return label_dict

    def __make_entity_dict(self, label_set: set) -> dict:
        """
        학습 등에 사용되는 엔티티 라벨 딕셔너리를 생성합니다.

        :param label_set: 엔티티 집합 (중복 X)
        :return: 엔티티 딕셔너리 (e.g. {B-DATE:1, B-LOCATION:2, ...})
        """

        label_set = sorted(list(label_set))
        label_dict = {}

        for i, entity in enumerate(label_set):
            label_dict[entity] = i  # 엔티티 딕셔너리 만듬

        return label_dict

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

    def __check_num_of_label(self, data_df: DataFrame) -> int:
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
                print(question, entity)
                number_of_error += 1

        if number_of_error != 0:
            raise Exception("THERE ARE {} ERRORS!\n".format(number_of_error))

        return number_of_error
