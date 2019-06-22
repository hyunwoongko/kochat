# Author : Hyunwoong
# When : 5/6/2019
# Homepage : github.com/gusdnd852
import os

from gensim.models import FastText
from konlpy.tag import Okt

from src.intent.classifier import intent_mapping
from src.intent.configs import IntentConfigs
from src.util.tokenizer import tokenize

configs = IntentConfigs()
data = configs.data
vector_size = configs.vector_size


def preprocess_data():
    data['intent'] = data['intent'].map(intent_mapping)
    count = 0
    for i in data['question']:
        data.replace(i, tokenize(i), regex=True, inplace=True)
        if count % 50 == 0:
            print("CURRENT COLLECT : ", count)
        count += 1

    encode = []
    decode = []
    for q, i in data.values:
        encode.append(q)
        decode.append(i)

    return {'encode': encode, 'decode': decode}


def train_vector_model(train_data_list, mode):
    if mode == 'train':
        mecab = Okt()
        str_buf = train_data_list['encode']
        joinString = ' '.join(str_buf)
        pos1 = mecab.pos(joinString)
        pos2 = ' '.join(list(map(lambda x: '\n' if x[1] in ['Punctuation'] else x[0], pos1))).split('\n')
        morphs = list(map(lambda x: mecab.morphs(x), pos2))
        print("BUILD MODEL")
        model = FastText(size=vector_size,
                         window=3,
                         workers=8,
                         min_count=1,
                         sg=1,
                         iter=1000)
        model.build_vocab(morphs)
        print("BUILD COMPLETE")

        print("TRAIN START")
        model.train(morphs, total_examples=model.corpus_count,
                    epochs=model.epochs,
                    compute_loss=True)
        if not os.path.exists('./fasttext'):
            os.makedirs('./fasttext')

        model.save('./fasttext/model')
        print("TRAIN COMPLETE")
        return model
    else:
        return FastText.load('./fasttext/model')
