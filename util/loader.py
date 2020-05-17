"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import pandas as pd

from util.tokenizer import Tokenizer


class TrainDataLoader:
    tok = Tokenizer()

    def load_intent(self, data_path):
        data = pd.read_csv(data_path)
        intent = data['intent']
        intent = intent.map(self.count_intent(intent))
        intent = intent.tolist()

        question = data['question']
        question = [self.tok.tokenize(i, train=True) for i in question]
        return {'data': question, 'label': intent}

    def load_intent_contrastive(self, data_path):
        data = pd.read_csv(data_path)
        intent = data['intent']
        intent = intent.map(self.count_intent(intent))
        intent = intent.tolist()

        question = data['question']
        question = [self.tok.tokenize(i, train=True) for i in question]

        return {'data': question, 'label': intent}

    def count_intent(self, label):
        count = {}
        index = -1

        for lb in label:
            if lb not in count:
                index += 1
            count[lb] = index
        return count

    def load_entity(self, data_path, label_path):
        data = self.read_line(open(data_path, mode='r', encoding='utf-8'))

        if label_path is not None:
            label = self.read_line(open(label_path, mode='r', encoding='utf-8'))
            return {'data': data, 'label': label}
        else:
            return {'data': data, 'label': None}

    def read_line(self, fp):
        all_line = []
        while True:
            line = fp.readline()
            if not line: break
            all_line.append(line.replace('\n', '').split(','))

        return all_line
