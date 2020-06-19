import os
import random

import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from backend.decorators import data


@data
class Dataset:

    def __init__(self, preprocessor, ood):
        self.__ood = ood
        self.__prep = preprocessor
        self.intent_dict = self.__prep.generate_intent()
        self.entity_dict = self.__prep.generate_entity()

    def load_embed(self):
        embed_dataset = pd.read_csv(self.intent_data_dir)

        if self.__ood:
            embed_dataset = pd.concat([embed_dataset, self.__read_ood()])

        embed_dataset = embed_dataset.values.tolist()
        embed_dataset = self.__tokenize_dataset(embed_dataset)
        return np.array(embed_dataset)[:, 0].tolist()
        # 0 : question, 1 : 라벨 / 임베딩에는 라벨 필요 없으니 버리기

    def load_intent(self, emb_processor):
        intent_dataset = pd.read_csv(self.intent_data_dir)
        intent_train, intent_test = self.__make_intent(intent_dataset, emb_processor)
        intent_train, intent_test = self.__mini_batch(intent_train), tuple(intent_test)

        if self.__ood:
            ood_dataset = self.__read_ood()
            ood_train, ood_test = self.__make_intent(ood_dataset, emb_processor)
            ood_train, ood_test = tuple(ood_train), tuple(ood_test)
            return intent_train, intent_test, ood_train, ood_test

        else:
            return intent_train, intent_test

    def load_entity(self, emb_processor):
        entity_dataset = pd.read_csv(self.entity_data_dir)
        entity_train, entity_test = self.__make_entity(entity_dataset, emb_processor)
        return self.__mini_batch(entity_train), tuple(entity_test)

    def load_predict(self, text, emb_processor):
        text = self.__prep.tokenize(text, train=False)

        if len(text) == 0:
            raise Exception("문장 길이가 0입니다.")

        text = emb_processor.inference(text)
        text, _ = self.__prep.pad_sequencing(text)
        return text.unsqueeze(0).to(self.device)

    def __make_intent(self, intent_dataset, emb_processor):
        intent_dataset = self.__map_label(intent_dataset, 'intent')
        intent_dataset = self.__tokenize_dataset(intent_dataset)
        train, test = self.__split_data(intent_dataset)

        train_question, train_label, train_length = self.__embedding(train, emb_processor)
        test_question, test_label, test_length = self.__embedding(test, emb_processor)

        train_tensors = self.__list2tensor(train_question, train_label, train_length)
        test_tensors = self.__list2tensor(test_question, test_label, test_length)
        return train_tensors, test_tensors

    def __make_entity(self, entity_dataset, emb_processor):
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
        ood_dataset = []
        for ood in os.listdir(self.ood_data_dir):
            if ood != '__init__.py':
                ood = pd.read_csv(self.ood_data_dir + ood)
                ood_dataset.append(ood)

        return pd.concat(ood_dataset)

    def __tokenize_dataset(self, dataset):
        return [[self.__prep.tokenize(q, train=True),
                 l if type(l) == list else [l]]
                for (q, l) in dataset]

    def __map_label(self, dataset, kinds):
        questions, labels = dataset['question'], None

        if kinds == 'intent':
            labels = dataset[kinds].map(self.intent_dict)
            labels.fillna(len(self.intent_dict), inplace=True)
            labels = labels.astype(int).tolist()

        elif kinds == 'entity':
            labels = [[self.entity_dict[e] for e in entity.split()]
                      for entity in dataset[kinds]]

        return list(zip(questions, labels))

    def __split_data(self, dataset):
        random.shuffle(dataset)
        split_point = int(len(dataset) * self.data_ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]
        return train_dataset, test_dataset

    def __embedding(self, dataset, emb_processor):
        question_list, label_list, length_list = [], [], []

        for i, (question, label) in enumerate(dataset):
            question = emb_processor.predict(question)
            question, length = self.__prep.pad_sequencing(question)

            question_list.append(question.unsqueeze(0))
            label_list.append(torch.tensor(label))
            length_list.append(torch.tensor(length).unsqueeze(0))

        return question_list, label_list, length_list

    def __list2tensor(self, *lists):
        return [torch.cat(a_list, dim=0) for a_list in lists]

    def __mini_batch(self, tensors):
        return DataLoader(TensorDataset(*tensors),
                          batch_size=self.batch_size,
                          shuffle=True)
