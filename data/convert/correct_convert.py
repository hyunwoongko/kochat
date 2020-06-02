import pandas as pd
import re

from util.tokenizer import Tokenizer


def convert():
    tok = Tokenizer()
    f = pd.read_csv('../intent/intent_여행지.csv', delimiter=',')
    sentence = f['question'].tolist()
    sentence_correct = []
    for s in sentence:
        correct = ' '.join(tok.tokenize(s))
        print(correct)
        sentence_correct.append(correct)

    intent = ['여행지' for _ in sentence]
    f = pd.DataFrame(data=zip(sentence, intent))
    f.to_csv('testfile_convert.csv', index=False)

if __name__ == '__main__':
    convert()
