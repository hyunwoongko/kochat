"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from config import Config
from data.data_builder import DataBuilder
from util.tokenizer import Tokenizer


class Dataset:

    def __init__(self):
        self.tok = Tokenizer()
        self.conf = Config()
        self.data_builder = DataBuilder(self.conf.raw_datapath)
        self.data_builder.build_intent()
        self.data_builder.build_entity()
        self.label_dict = {}
        # label mapping dictionary

    def embed_train(self):
        dataset = pd.read_csv(self.conf.intent_datapath)
        label = dataset['intent']
        label = label.map(self.__count_intent(label)).tolist()
        data = [self.tok.tokenize(i, train=True) for i in dataset['question']]
        return {'data': data, 'label': label}

    def intent_train(self, emb):
        dataset = self.embed_train()  # TOKENIZING HERE
        data, label = dataset['data'], dataset['label']
        dataset = [zipped for zipped in zip(data, label)]
        print("INTENT : ZIPPING DONE")
        return self.make_dataset(emb, dataset, self.conf.intent_ratio)

    def entity_train(self, emb):
        label_set = self.data_builder.entity_set
        label_set = list(label_set)
        label_set = sorted(label_set)
        # 항상 동일한 결과를 보여주려면 정렬해놔야 함

        for i, entity in enumerate(label_set):
            self.label_dict[entity] = i
        print("ENTITY : MAKING LABEL DICT DONE")

        entity_dataset = pd.read_csv(self.conf.entity_datapath)
        entity_questionset = entity_dataset['question'].values.tolist()
        entity_questionset = [self.tok.tokenize(q, train=True) for q in entity_questionset]
        entity_labelset = []

        for entity in entity_dataset['entity']:
            entity = entity.split()
            entity = [self.label_dict[e] for e in entity]
            entity_labelset.append(entity)
        print("ENTITY : LABEL MAPPING DONE")
        print(self.label_dict)

        entity_dataset = []
        for data in zip(entity_questionset, entity_labelset):
            question, entity = data[0], data[1]
            data_row = [question, entity]
            entity_dataset.append(data_row)
        print("ENTITY : ZIPPING DONE")

        return self.make_dataset(emb, entity_dataset, self.conf.entity_ratio)

    def make_dataset(self, emb, dataset, ratio):
        # 1. split data to train / test
        random.shuffle(dataset)
        split_point = int(len(dataset) * ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]
        print("INTENT : CUTTING DONE")

        # 2. do embedding & pad sequencing
        train_embedded, train_label = self.__embedding(emb, train_dataset)
        test_embedded, test_label = self.__embedding(emb, test_dataset)
        print("INTENT : EMBEDDING DONE")

        # 3. concatenate list → torch.tensor
        train_dataset, test_dataset = \
            torch.cat(train_embedded, dim=0), torch.cat(test_embedded, dim=0)

        train_label, test_label = \
            torch.cat(train_label, dim=0), torch.cat(test_label, dim=0)
        print("INTENT : CONCATENATING DONE")

        # 4. make mini batch
        train_set = TensorDataset(train_dataset, train_label)
        train_set = DataLoader(train_set, batch_size=self.conf.batch_size, shuffle=True)
        test_set = (test_dataset, test_label)  # for onetime test
        print("INTENT : MINI BATCH DONE")
        return train_set, test_set

    def __embedding(self, emb, input_dataset):
        embedded_list, label_list = [], []
        for i, (input_data, input_label) in enumerate(input_dataset):
            input_data = self.__pad_sequencing(emb.embed(input_data))
            embedded_list.append(input_data.unsqueeze(0))

            input_label = torch.tensor(input_label)
            if len(input_label.size()) == 0:
                # INTENT의 경우 라벨의 차원이 0(스칼라)임
                # e.g. (0) or (3) or (2)...
                label_list.append(input_label.unsqueeze(0))
            else:
                # ENTITY의 경우 라벨의 차원이 1(벡터)임
                # 길이가 모두 다르기 때문에, pad sequencing 필요
                # e.g. (LOCATION, O, O) or (O, O, DATE, O) ...
                if input_label.size()[0] > self.conf.max_len:
                    input_label = input_label[:self.conf.max_len]
                else:
                    while input_label.size()[0] != self.conf.max_len:
                        non_tag = self.label_dict[self.conf.non_tag]
                        non_tag_tensor = torch.tensor(non_tag).unsqueeze(0)
                        input_label = torch.cat([input_label, non_tag_tensor])

                label_list.append(input_label.unsqueeze(0))

        return embedded_list, label_list

    def __pad_sequencing(self, sequence):
        if sequence.size()[0] > self.conf.max_len:
            sequence = sequence[:self.conf.max_len]
        else:
            pad = torch.zeros(self.conf.max_len, self.conf.vector_size)
            for i in range(sequence.size()[0]):
                pad[i] = sequence[i]
            sequence = pad

        return sequence

    @staticmethod
    def __count_intent(label):
        count, index = {}, -1
        for lb in label:
            if lb not in count:
                index += 1
            count[lb] = index
        return count
