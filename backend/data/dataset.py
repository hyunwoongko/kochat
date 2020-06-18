import os
import random

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataset import TensorDataset as TorchDataset

from backend.decorators import data


@data
class Dataset:
    def __init__(self, preprocessor, ood):
        self.ood = ood
        self.__preprocessor = preprocessor
        self.__preprocessor.generate_intent()
        self.__preprocessor.generate_entity()

        i, e = self.__preprocessor.make_dicts()
        self.intent_dict = i
        self.entity_dict = e

    def embed_dataset(self):
        embed_dataset = pd.read_csv(self.intent_data_dir)
        if self.ood:
            ood_dataset = []
            for ood in os.listdir(self.ood_data_dir):
                if ood != '__init__.py':
                    ood = pd.read_csv(self.ood_data_dir + ood)
                    ood_dataset.append(ood)

            ood_dataset = pd.concat(ood_dataset)
            embed_dataset = pd.concat([ood_dataset, embed_dataset])

        embed_dataset = embed_dataset['question']
        embed_dataset = [self.__preprocessor.tokenize(i, train=True)
                         for i in embed_dataset]

        return embed_dataset

    def intent_dataset(self, emb):
        dataset = pd.read_csv(self.intent_data_dir)
        label = dataset['intent']
        label = label.map(self.intent_dict).tolist()
        data = [self.__preprocessor.tokenize(i, train=True)
                for i in dataset['question']]
        dataset = [zipped for zipped in zip(data, label)]

        train_set, test_set = self.__make_dataset(emb, dataset)
        if not self.ood:
            return train_set, test_set

        else:
            return train_set, test_set, self.__ood_dataset(emb)

    def entity_dataset(self, emb):
        entity_dataset = pd.read_csv(self.entity_data_dir)
        entity_questionset = entity_dataset['question'].values.tolist()
        entity_questionset = [self.__preprocessor.tokenize(q, train=True)
                              for q in entity_questionset]
        entity_labelset = [[self.entity_dict[e] for e in entity.split()]
                           for entity in entity_dataset['entity']]
        entity_dataset = list(zip(entity_questionset, entity_labelset))
        return self.__make_dataset(emb, entity_dataset)

    def inference_dataset(self, text, emb):
        text = self.__preprocessor.tokenize(text, train=False)
        if len(text) == 0:
            raise Exception("문장 길이가 0입니다.")
        text = emb.inference(text)
        text = self.__preprocessor.pad_sequencing(text)
        text = text.unsqueeze(0).to(self.device)
        return text

    def __ood_dataset(self, emb):
        dataset, i = [], 0
        for ood in os.listdir(self.ood_data_dir):
            if ood != '__init__.py':
                ood = pd.read_csv(self.ood_data_dir + ood)
                ood['intent'] = i + len(self.intent_dict)
                i += 1  # ood_close와 ood_open 구분
                dataset.append(ood)

        dataset = pd.concat(dataset)
        data = [self.__preprocessor.tokenize(i, train=True)
                for i in dataset['question']]
        label = dataset['intent']
        dataset = [zipped for zipped in zip(data, label)]

        random.shuffle(dataset)
        split_point = int(len(dataset) * self.data_ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]

        train_embedded, train_label = self.__embedding(emb, train_dataset)
        test_embedded, test_label = self.__embedding(emb, test_dataset)

        train_dataset = torch.cat(train_embedded, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_dataset = torch.cat(test_embedded, dim=0)
        test_label = torch.cat(test_label, dim=0)

        train_dataset = (train_dataset, train_label)
        test_dataset = (test_dataset, test_label)
        return train_dataset, test_dataset

    def __make_dataset(self, emb, dataset):
        # 1. split data to train / test
        random.shuffle(dataset)
        split_point = int(len(dataset) * self.data_ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]

        # 2. do embedding & pad sequencing
        train_embedded, train_label = self.__embedding(emb, train_dataset)
        test_embedded, test_label = self.__embedding(emb, test_dataset)

        # 3. concatenate list → torch.tensor
        train_dataset, test_dataset = \
            torch.cat(train_embedded, dim=0), torch.cat(test_embedded, dim=0)

        train_label, test_label = \
            torch.cat(train_label, dim=0), torch.cat(test_label, dim=0)

        # 4. make mini batch
        train_set = TorchDataset(train_dataset, train_label)
        train_set = TorchDataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_set = (test_dataset, test_label)  # for onetime test
        return train_set, test_set

    def __embedding(self, emb, input_dataset):
        embedded_list, label_list = [], []
        for i, (input_data, input_label) in enumerate(input_dataset):
            input_data = self.__preprocessor.pad_sequencing(emb.inference(input_data))
            embedded_list.append(input_data.unsqueeze(0))
            input_label = torch.tensor(input_label)

            # INTENT는 라벨의 차원이 0(스칼라)인데 반해 ENTITY의 경우 라벨의 차원이 1(벡터)임.
            # 라벨도 길이가 모두 다르기 때문에 pad sequencing 필요 (e.g. 길이=4, (DATE, O) → (DATE, O, O, O))
            if len(input_label.size()) != 0:
                if input_label.size()[0] > self.max_len:
                    input_label = input_label[:self.max_len]
                else:
                    while input_label.size()[0] != self.max_len:
                        outside_tag = self.entity_dict[self.NER_outside]
                        outside_tag = torch.tensor(outside_tag).unsqueeze(0)
                        input_label = torch.cat([input_label, outside_tag])

            label_list.append(input_label.unsqueeze(0))
        return embedded_list, label_list
