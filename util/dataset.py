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
from util.tokenizer import Tokenizer


class Dataset:
    tok = Tokenizer()
    conf = Config()

    def embed_train(self, data_path):
        dataset = pd.read_csv(data_path)
        label = dataset['intent']
        label = label.map(self.count_intent(label)).tolist()
        data = dataset['question']
        data = [self.tok.tokenize(i, train=True) for i in data]
        return {'data': data, 'label': label}

    def intent_embedding(self, emb, input_dataset):
        embedded_list, label_list = [], []
        for i, (input_data, input_label) in enumerate(input_dataset):
            input_data = self.pad_sequencing(emb.embed(input_data))
            embedded_list.append(input_data.unsqueeze(0))
            label_list.append(torch.tensor(input_label).unsqueeze(0))
            print("INTENT : EMBEDDING : ", (i / len(input_dataset)) * 100, "%")

        return embedded_list, label_list

    def siamese_embedding(self, emb, input_dataset, input_label):
        embedded, label = [], []
        for data in input_dataset:
            d1, d2 = data[0], data[1]
            d1 = self.pad_sequencing(emb.embed(d1)).unsqueeze(0)
            d2 = self.pad_sequencing(emb.embed(d2)).unsqueeze(0)
            embedded.append(torch.cat([d1, d2], dim=0).unsqueeze(0))
            label.append(torch.tensor(input_label).unsqueeze(0))  # pos : 1 / neg : 0
        return embedded, label

    def intent_train(self, emb, data_path):
        # 1. load data from csv files and tokenizing
        dataset = self.embed_train(data_path)  # TOKENIZING HERE
        print("INTENT : SPITING DONE")
        data, label = dataset['data'], dataset['label']
        dataset = [zipped for zipped in zip(data, label)]
        print("INTENT : ZIPPING DONE")

        # 2. split data to train / test
        random.shuffle(dataset)
        split_point = int(len(dataset) * self.conf.intent_ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]
        print("INTENT : CUTTING DONE")

        # 3. do embedding
        train_embedded, train_label = self.intent_embedding(emb, train_dataset)
        test_embedded, test_label = self.intent_embedding(emb, test_dataset)
        print("INTENT : EMBEDDING DONE")

        # 4. concatenate list → torch.tensor
        train_dataset, test_dataset = torch.cat(train_embedded, dim=0), torch.cat(test_embedded, dim=0)
        train_label, test_label = torch.cat(train_label, dim=0), torch.cat(test_label, dim=0)
        print("INTENT : CONCATENATING DONE")

        # 5. make mini batch
        train_set = TensorDataset(train_dataset, train_label)
        train_set = DataLoader(train_set, batch_size=self.conf.batch_size, shuffle=True)
        test_set = (test_dataset, test_label)  # for onetime test
        print("INTENT : MINI BATCH DONE")

        return train_set, test_set

    def siamese_train(self, emb, data_path):
        # 1. load data from csv files
        print("SIAMESE : TOKENIZING")
        dataset = self.embed_train(data_path)
        data, label = dataset['data'], dataset['label']
        dataset = [zipped for zipped in zip(data, label)]

        # 2. split data to train / test
        print("SIAMESE : SPITING")
        random.shuffle(dataset)
        split_point = int(len(dataset) * self.conf.intent_ratio)
        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]

        # 3. make pairwise dataset
        print("SIAMESE : PAIRWISE")
        train_dataset = self.make_even_number_data(train_dataset)
        test_dataset = self.make_even_number_data(test_dataset)
        train_pos_pair, train_neg_pair = self.make_pos_neg_pair(train_dataset)
        test_pos_pair, test_neg_pair = self.make_pos_neg_pair(test_dataset)

        # # 4. do embedding
        print("SIAMESE : EMBEDDING")
        train_embedded_pos, train_label_pos = self.siamese_embedding(emb, train_pos_pair, input_label=1)
        train_embedded_neg, train_label_neg = self.siamese_embedding(emb, train_neg_pair, input_label=0)
        test_embedded_pos, test_label_pos = self.siamese_embedding(emb, test_pos_pair, input_label=1)
        test_embedded_neg, test_label_neg = self.siamese_embedding(emb, test_neg_pair, input_label=0)
        train_embedded = train_embedded_pos + train_embedded_neg
        train_label = train_label_pos + train_label_neg
        test_embedded = test_embedded_pos + test_embedded_neg
        test_label = test_label_pos + test_label_neg

        # # 5. concatenate and make mini batch
        print("SIAMESE : MAKE DATASET")
        train_dataset, test_dataset = torch.cat(train_embedded, dim=0), torch.cat(test_embedded, dim=0)
        train_label, test_label = torch.cat(train_label, dim=0), torch.cat(test_label, dim=0)
        train_set = DataLoader(TensorDataset(train_dataset, train_label),
                               batch_size=self.conf.batch_size, shuffle=True)
        test_set = DataLoader(TensorDataset(test_dataset, test_label),
                              batch_size=self.conf.batch_size, shuffle=True)
        return train_set, test_set

    def count_intent(self, label):
        count, index = {}, -1
        for lb in label:
            if lb not in count:
                index += 1
            count[lb] = index
        return count

    def read_line(self, fp):
        all_line = []
        while True:
            line = fp.readline()
            if not line: break
            all_line.append(line.replace('\n', '').split(','))

        return all_line

    def pad_sequencing(self, sequence):
        if sequence.size()[0] > self.conf.max_len:
            sequence = sequence[:self.conf.max_len]
        else:
            pad = torch.zeros(self.conf.max_len, self.conf.vector_size)
            for i in range(sequence.size()[0]):
                pad[i] = sequence[i]
            sequence = pad

        return sequence

    def make_even_number_data(self, input_dataset):
        if len(input_dataset) % 2 != 0:
            del input_dataset[len(input_dataset) - 1]
        return input_dataset

    def make_pos_neg_pair(self, input_dataset):
        pos, neg, cache = [], [], None
        for i, d in enumerate(input_dataset):
            if i != 0:
                if d[1] == cache[1]:
                    pos.append((d[0], cache[0]))
                else:
                    neg.append((d[0], cache[0]))
            cache = d

        return pos, neg
