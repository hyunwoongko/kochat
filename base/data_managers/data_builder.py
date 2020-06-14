import random
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from base.data_managers.data_generator import DataGenerator


class DataBuilder(DataGenerator):
    def __init__(self):
        super().__init__()
        self._generate_intent()
        self._generate_entity()
        self.entity_dict = {}
        self.intent_dict = {}

    def inference_sequence(self, text, emb):
        text = self.tokenize(text, train=False)
        text = emb.embed(text)
        text = self.pad_sequencing(text)
        text = text.unsqueeze(0).cuda()
        return text

    def embed_dataset(self):
        dataset = pd.read_csv(self.intent_data_file)
        label = dataset['intent']
        count, index = {}, -1

        # label to int
        for lb in label:
            if lb not in count:
                index += 1
            count[lb] = index

        self.intent_dict = count
        label = label.map(self.intent_dict).tolist()
        data = [self.tokenize(i, train=True)
                for i in dataset['question']]

        return data, label

    def intent_dataset(self, emb):
        data, label = self.embed_dataset()
        dataset = [zipped for zipped in zip(data, label)]
        print("DATA_BUILDER : ZIPPING DONE")
        return self._make_dataset(emb, dataset, self.data_ratio)

    def entity_dataset(self, emb):
        label_set = self.entity_set
        label_set = list(label_set)
        label_set = sorted(label_set)
        # 항상 동일한 결과를 보여주려면 정렬해놔야 함

        for i, entity in enumerate(label_set):
            self.entity_dict[entity] = i
        print("DATA_BUILDER : MAKING LABEL DICT DONE")

        entity_dataset = pd.read_csv(self.entity_data_file)
        entity_questionset = entity_dataset['question'].values.tolist()
        entity_questionset = [self.tokenize(q, train=True)
                              for q in entity_questionset]

        entity_labelset = []
        for entity in entity_dataset['entity']:
            entity = entity.split()
            entity = [self.entity_dict[e] for e in entity]
            entity_labelset.append(entity)
        print("DATA_BUILDER : LABEL MAPPING DONE")

        entity_dataset = []
        for data in zip(entity_questionset, entity_labelset):
            question, entity = data[0], data[1]
            data_row = [question, entity]
            entity_dataset.append(data_row)
        print("DATA_BUILDER : ZIPPING DONE")
        return self._make_dataset(emb, entity_dataset, self.data_ratio)

    def _make_dataset(self, emb, dataset, ratio):
        # 1. split data to train / test
        random.shuffle(dataset)
        split_point = int(len(dataset) * ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]
        print("DATA_BUILDER : CUTTING DONE")

        # 2. do embedding & pad sequencing
        train_embedded, train_label = self._embedding(emb, train_dataset)
        test_embedded, test_label = self._embedding(emb, test_dataset)
        print("DATA_BUILDER : EMBEDDING DONE")

        # 3. concatenate list → torch.tensor
        train_dataset, test_dataset = \
            torch.cat(train_embedded, dim=0), torch.cat(test_embedded, dim=0)

        train_label, test_label = \
            torch.cat(train_label, dim=0), torch.cat(test_label, dim=0)
        print("DATA_BUILDER : CONCATENATING DONE")

        # 4. make mini batch
        train_set = TensorDataset(train_dataset, train_label)
        train_set = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_set = (test_dataset, test_label)  # for onetime test
        print("DATA_BUILDER : MINI BATCH DONE")
        return train_set, test_set

    def _embedding(self, emb, input_dataset):
        embedded_list, label_list = [], []
        for i, (input_data, input_label) in enumerate(input_dataset):
            input_data = self.pad_sequencing(emb.embed(input_data))
            embedded_list.append(input_data.unsqueeze(0))

            input_label = torch.tensor(input_label)
            if len(input_label.size()) != 0:
                # INTENT는 라벨의 차원이 0(스칼라)인데 반해
                # ENTITY의 경우 라벨의 차원이 1(벡터)임.
                # 길이가 모두 다르기 때문에, pad sequencing 필요
                # e.g. (LOCATION, O, O) or (O, O, DATE, O) ...
                if input_label.size()[0] > self.max_len:
                    input_label = input_label[:self.max_len]
                else:
                    while input_label.size()[0] != self.max_len:
                        non_tag = self.entity_dict[self.NER_outside]
                        non_tag_tensor = torch.tensor(non_tag).unsqueeze(0)
                        input_label = torch.cat([input_label, non_tag_tensor])

            label_list.append(input_label.unsqueeze(0))

        return embedded_list, label_list
