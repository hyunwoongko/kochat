import json
import os
import random
import re
import pandas as pd
import torch
from konlpy.tag import Okt
from requests import Session
from backend.data.preprocessor import Preprocessor
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataset import TensorDataset as TorchDataset


class Dataset(Preprocessor):
    def __init__(self):
        super().__init__()
        self._okt = Okt()
        self._generate_intent()
        self._generate_entity()

        i, e = self._make_dicts()
        self.intent_dict = i
        self.entity_dict = e

    def embed_dataset(self):
        ood_dataset = []
        for ood in os.listdir(self.ood_data_dir):
            if ood != '__init__.py':
                ood = pd.read_csv(self.ood_data_dir + ood)
                ood_dataset.append(ood)

        ood_dataset = pd.concat(ood_dataset)
        intent_dataset = pd.read_csv(self.intent_data_dir)
        embed_dataset = pd.concat([ood_dataset, intent_dataset])
        embed_dataset = embed_dataset['question']
        embed_dataset = [self.tokenize(i, train=True)
                         for i in embed_dataset]

        return embed_dataset

    def intent_dataset(self, emb):
        dataset = pd.read_csv(self.intent_data_dir)
        label = dataset['intent']
        label = label.map(self.intent_dict).tolist()
        data = [self.tokenize(i, train=True)
                for i in dataset['question']]
        dataset = [zipped for zipped in zip(data, label)]
        return self._make_dataset(emb, dataset, self.data_ratio)

    def entity_dataset(self, emb):
        entity_dataset = pd.read_csv(self.entity_data_dir)
        entity_questionset = entity_dataset['question'].values.tolist()
        entity_questionset = [self.tokenize(q, train=True)
                              for q in entity_questionset]
        entity_labelset = [[self.entity_dict[e] for e in entity.split()]
                           for entity in entity_dataset['entity']]
        entity_dataset = list(zip(entity_questionset, entity_labelset))
        return self._make_dataset(emb, entity_dataset, self.data_ratio)

    def inference_dataset(self, text, emb):
        text = self.tokenize(text, train=False)
        text = emb.inference(text)
        text = self.pad_sequencing(text)
        text = text.unsqueeze(0).to(self.device)
        return text

    def ood_dataset(self, emb):
        ood_dataset = []
        for ood in os.listdir(self.ood_data_dir):
            if ood != '__init__.py':
                ood = pd.read_csv(self.ood_data_dir + ood)
                ood_dataset.append(ood)

        ood_dataset = pd.concat(ood_dataset)
        intent_dataset = pd.read_csv(self.intent_data_dir)
        embed_dataset = pd.concat([ood_dataset, intent_dataset])
        embed_dataset = embed_dataset['question']
        embed_dataset = [self.tokenize(i, train=True)
                         for i in embed_dataset]

        return embed_dataset

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
            sentence = self._naver_fix(sentence)
            sentence = self._okt.pos(sentence)
            out = [word for word, pos in sentence
                   if pos not in ['Josa', 'Punctuation']]

            return self._naver_fix(' '.join(out)).split()

    def _naver_fix(self, text):
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

    def _make_dicts(self):
        intent_dict, entity_dict = {}, {}
        dataset = pd.read_csv(self.intent_data_dir)
        label, index = dataset['intent'], -1

        # label to int
        for lb in label:
            if lb not in intent_dict:
                index += 1
            intent_dict[lb] = index

        label_set = self._entity_set
        label_set = list(label_set)
        label_set = sorted(label_set)
        # 항상 동일한 결과를 보여주려면 정렬해놔야 함

        for i, entity in enumerate(label_set):
            entity_dict[entity] = i

        return intent_dict, entity_dict

    def _make_dataset(self, emb, dataset, ratio):
        # 1. split data to train / test
        random.shuffle(dataset)
        split_point = int(len(dataset) * ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]

        # 2. do embedding & pad sequencing
        train_embedded, train_label = self._embedding(emb, train_dataset)
        test_embedded, test_label = self._embedding(emb, test_dataset)

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

    def _embedding(self, emb, input_dataset):
        embedded_list, label_list = [], []
        for i, (input_data, input_label) in enumerate(input_dataset):
            input_data = self.pad_sequencing(emb.inference(input_data))
            embedded_list.append(input_data.unsqueeze(0))

            input_label = torch.tensor(input_label)
            if len(input_label.size()) != 0:
                # INTENT는 라벨의 차원이 0(스칼라)인데 반해
                # ENTITY의 경우 라벨의 차원이 1(벡터)임.
                # 길이가 모두 다르기 때문에, 라벨도 pad sequencing 필요
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
