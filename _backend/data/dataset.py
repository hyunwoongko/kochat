import os
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from _backend.data.preprocessor import Preprocessor
from _backend.decorators import data
from _backend.proc.base.base_processor import BaseProcessor


@data
class Dataset:
    """
    Class Dataset : @data (DATA configuration)

    학습과 추론에 사용할 데이터셋을 생성하는 클래스입니다.
    """

    def __init__(self, preprocessor: Preprocessor, ood: bool):
        """
        전처리기는 현재는 한 가지만 구현되어 있으며, 추후 추가될 수 있습니다.
        Preprocessor()로 생성하여 입력하면 되고, ood는 Out of distribution 데이터셋 사용 여부입니다.
        ood 데이터를 쓰면 Threshold 설정 없이 Automatic Fallback Detection이 가능합니다.

        :param preprocessor: 전처리기 객체(Preprocessor())입니다.
        :param ood: Out of distribution dataset 사용 여부입니다.
        """

        self.__ood = ood
        self.__prep = preprocessor
        self.intent_dict = self.__prep.generate_intent()
        self.entity_dict = self.__prep.generate_entity()

    def load_embed(self) -> list:
        """
        임베딩 프로세서 학습용 데이터를 생성합니다.
        임베딩 데이터셋은 라벨(인텐트, 엔티티)가 필요하지 않기 때문에
        라벨을 자르고 자연어 데이터 부분만 반환합니다.

        :return: 라벨이 없는 임베딩 학습용 데이터셋입니다.
        """

        embed_dataset = pd.read_csv(self.intent_data_dir)

        if self.__ood:
            embed_dataset = pd.concat([embed_dataset, self.__read_ood()])

        embed_dataset = embed_dataset.values.tolist()
        embed_dataset = self.__tokenize_dataset(embed_dataset)
        question_index, label_index = 0, 1  # 매직넘버 사용하지 않기 !
        return np.array(embed_dataset)[:, question_index].tolist()

    def load_intent(self, emb_processor: BaseProcessor) -> tuple:
        """
        인텐트 프로세서 학습용 데이터를 생성합니다.
        :param emb_processor: 임베딩 과정이 들어가므로 임베딩 프로세서를 입력해야합니다.

        :return: 인텐트 프로세서 학습용 데이터셋입니다.
        """

        intent_dataset = pd.read_csv(self.intent_data_dir)
        intent_train, intent_test = self.__make_intent(intent_dataset, emb_processor)
        intent_train, intent_test = self.__mini_batch(intent_train), tuple(intent_test)
        # 테스트는 여러 에폭을 거치지 않고 마지막에만 한번 수행되기 때문에 tuple로 리턴합니다.

        if self.__ood:
            ood_dataset = self.__read_ood()
            ood_train, ood_test = self.__make_intent(ood_dataset, emb_processor)
            ood_train, ood_test = tuple(ood_train), tuple(ood_test)
            return intent_train, intent_test, ood_train, ood_test

        else:
            return intent_train, intent_test

    def load_entity(self, emb_processor: BaseProcessor) -> tuple:
        """
        엔티티 프로세서 학습용 데이터를 생성합니다.
        :param emb_processor: 임베딩 과정이 들어가므로 임베딩 프로세서를 입력해야합니다.

        :return: 엔티티 프로세서 학습용 데이터셋입니다.
        """

        entity_dataset = pd.read_csv(self.entity_data_dir)
        entity_train, entity_test = self.__make_entity(entity_dataset, emb_processor)
        return self.__mini_batch(entity_train), tuple(entity_test)

    def load_predict(self, text: str, emb_processor: BaseProcessor) -> torch.Tensor:
        """
        실제 애플리케이션 등에서 유저 입력에 대한 인퍼런스를 수행할 때
        사용자가 입력한 Raw 텍스트(str)를 텐서로 변환합니다.

        :param text: 사용자의 텍스트 입력입니다.
        :param emb_processor: 임베딩 과정이 들어가므로 임베딩 프로세서를 입력해야합니다.
        :return: 유저 입력 추론용 텐서를 리턴합니다.
        """

        text = self.__prep.tokenize(text, train=False)

        if len(text) == 0:
            raise Exception("문장 길이가 0입니다.")

        text = emb_processor.predict(text)
        text, _ = self.__prep.pad_sequencing(text)
        return text.unsqueeze(0).to(self.device)

    def __make_intent(self, intent_dataset, emb_processor) -> tuple:
        """
        인텐트 데이터셋을 만드는 세부 과정입니다.

        1. 라벨을 숫자로 맵핑합니다.
        2. 데이터를 토큰화 합니다 (네이버 맞춤법 검사기 + Konlpy 사용)
        3. 데이터를 학습 / 검증용으로 나눕니다.
        4. 데이터의 길이를 맞추기 위해 패드시퀀싱 후 임베딩합니다.
        5. 리스트로 출력된 데이터들을 concatenation하여 텐서로 변환합니다.

        :param intent_dataset: 저장공간에서 로드한 인텐트 데이터 파일입니다.
        :param emb_processor: 임베딩을 위한 임베딩 프로세서를 입력해야합니다.
        :return: 텐서로 변환된 인텐트 데이터입니다.
        """

        intent_dataset = self.__map_label(intent_dataset, 'intent')
        intent_dataset = self.__tokenize_dataset(intent_dataset)
        train, test = self.__split_data(intent_dataset)

        train_question, train_label, train_length = self.__embedding(train, emb_processor)
        test_question, test_label, test_length = self.__embedding(test, emb_processor)

        train_tensors = self.__list2tensor(train_question, train_label, train_length)
        test_tensors = self.__list2tensor(test_question, test_label, test_length)
        return train_tensors, test_tensors

    def __make_entity(self, entity_dataset, emb_processor) -> tuple:
        """
        엔티티 데이터셋을 만드는 세부 과정입니다.
        
        1. 라벨을 숫자로 맵핑합니다.
        2. 데이터를 토큰화 합니다 (네이버 맞춤법 검사기 + Konlpy 사용)
        3. 데이터를 학습 / 검증용으로 나눕니다.
        4. 데이터의 길이를 맞추기 위해 패드시퀀싱 후 임베딩합니다.
        5. 엔티티 데이터는 라벨도 각각 길이가 달라서 패드시퀀싱 해야합니다.
        6. 리스트로 출력된 데이터들을 concatenation하여 텐서로 변환합니다.
        
        :param entity_dataset: 저장공간에서 로드한 엔티티 데이터 파일입니다.
        :param emb_processor: 임베딩을 위한 임베딩 프로세서를 입력해야합니다.
        :return: 텐서로 변환된 엔티티 데이터입니다.
        """

        entity_dataset = self.__map_label(entity_dataset, 'entity')
        entity_dataset = self.__tokenize_dataset(entity_dataset)
        train, test = self.__split_data(entity_dataset)

        train_question, train_label, train_length = self.__embedding(train, emb_processor)
        test_question, test_label, test_length = self.__embedding(train, emb_processor)

        train_label = [self.__prep.label_sequencing(label, self.entity_dict) for label in train_label]
        test_label = [self.__prep.label_sequencing(label, self.entity_dict) for label in test_label]

        train_tensors = self.__list2tensor(train_question, train_label, train_length)
        test_tensors = self.__list2tensor(test_question, test_label, test_length)
        return train_tensors, test_tensors

    def __read_ood(self):
        """
        메모리에서 OOD 데이터를 로드합니다.
        OOD 데이터는 폴백 디텍션 모델의 Threshold를 자동으로 설정하고,
        인텐트 검색기와 폴백 디텍션 모델의 성능을 검증하기 위해 사용됩니다.

        :return: 여러개의 OOD 데이터를 한 파일로 모아서 반환합니다.
        """

        ood_dataset = []
        for ood in os.listdir(self.ood_data_dir):
            if ood != '__init__.py':
                ood = pd.read_csv(self.ood_data_dir + ood)
                ood_dataset.append(ood)

        return pd.concat(ood_dataset)

    def __map_label(self, dataset, kinds) -> list:
        """
        1단계 : 라벨 맵핑

        라벨을 맵핑합니다. 데이터를 처음 불러오면 라벨이 자연어이기 때문에
        이를 컴퓨터가 이해 가능한 숫자의 형태로 맵핑합니다.

        :param dataset: 메모리로 부터 불러와진 데이터셋입니다.
        :param kinds: 어떤 종류의 데이터인지(intent or entity)를 나타냅니다.
        :return: 맵핑이 완료된 리스트 데이터 셋
        """

        questions, labels = dataset['question'], None

        if kinds == 'intent':
            labels = dataset[kinds].map(self.intent_dict)
            labels.fillna(len(self.intent_dict), inplace=True)
            # ood 파일의 경우 intent dict에 라벨이 없기 때문에 nan가 됨.
            labels = labels.astype(int).tolist()
            # nan fillna하면 float이 되기 때문에 이를 int로 바꿔줌

        elif kinds == 'entity':
            labels = [[self.entity_dict[e] for e in entity.split()]
                      for entity in dataset[kinds]]

        return list(zip(questions, labels))

    def __tokenize_dataset(self, dataset: list):
        """
        2단계 : 데이터 토큰화
        
        데이터를 토큰화 합니다.
        엔티티 데이터의 경우 라벨이 1차원 리스트인데 반해,
        인텐트 데이터는 라벨이 Scalar 이므로 만약 list가 아닌 경우는
        unsqueeze하여 리스트로 만듭니다. (for concatenation !)

        :param dataset: 라벨 매핑이 완료된 리스트 데이터셋입니다.
        :return: 토큰화가 완료된 리스트 데이터셋입니다.
        """

        return [[self.__prep.tokenize(q, train=True),
                 l if type(l) == list else [l]] for (q, l) in dataset]

    def __split_data(self, dataset: list) -> tuple:
        """
        3단계 : 학습용/검증용 데이터 분리
        
        데이터셋을 학습용 / 검증용으로 나눕니다.
        Configuration에 적힌 split ratio를 기준으로 데이터를 쪼갭니다.

        :param dataset: 토큰화가 완료된 리스트 데이터셋
        :return: 분리가 완료된 (학습용 데이터, 검증용 데이터)
        """

        random.shuffle(dataset)
        split_point = int(len(dataset) * self.data_ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]
        return train_dataset, test_dataset

    def __embedding(self, dataset: list, emb_processor: BaseProcessor):
        """
        4단계 : 데이터 임베딩

        자연어 데이터를 연산하기 위해 임베딩합니다.
        임베딩 프로세서를 입력해야하며,
        임베딩을 위해서 각각 다른 데이터들의 길이를 일정하게 맞추고,
        (Pad Sequencing 진행) 임베딩 프로세서로 임베딩합니다.


        :param dataset: 이전단계에서 학습/검증용으로 나뉜 데이터 중 하나
        :param emb_processor: 임베딩을 위한 임베딩 프로세서
        :return: 임베딩된 자연어 데이터, 라벨 데이터, 길이 데이터
        """

        question_list, label_list, length_list = [], [], []

        for i, (question, label) in enumerate(dataset):
            question = emb_processor.predict(question)
            question, length = self.__prep.pad_sequencing(question)

            question_list.append(question.unsqueeze(0))
            label_list.append(torch.tensor(label))
            length_list.append(torch.tensor(length).unsqueeze(0))

        return question_list, label_list, length_list

    # torch.Tensor로 unpacking 됩니다.
    def __list2tensor(self, *lists: list) -> list:
        """
        5단계 : 데이터 병합 (텐서 변환)

        리스트 데이터들을 Pytorch의 Tensor 형태로 변환하여 병합합니다.
        각 리스트는 0번 Axis로 모두 Unsqueeze 되어 있어야 합니다.

        :param lists: 텐서로 만들 리스트 데이터셋들
        :return: 텐서 데이터들이 담긴 리스트 (각각 텐서로 unpacking 됩니다)
        """

        return [torch.cat(a_list, dim=0) for a_list in lists]

    def __mini_batch(self, tensors: tuple) -> DataLoader:
        """
        6단계 : 미니배치 형태로 자르기

        데이터를 미니배치 형태로 쪼개서 로딩할 수 있게 하는
        Pytorch DataLoader로 만듭니다.

        :param tensors: 텐서로 병합한 데이터셋들
        :return: 미니배치 트레이닝용 데이터로더 객체
        """

        return DataLoader(TensorDataset(*tensors),
                          batch_size=self.batch_size,
                          shuffle=True)
