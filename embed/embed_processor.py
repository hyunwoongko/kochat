"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
import torch
from gensim.models import FastText

from config import Config
from embed.embed_callback import EmbedCallback
from util.dataset import Dataset
from util.tokenizer import Tokenizer


class EmbedProcessor:
    tok = Tokenizer()
    conf = Config()
    data = Dataset()
    model = None

    def train(self):
        print('FAST_TEXT : start to make dataset')
        dataset = self.data.embed_train(self.conf.intent_datapath)['data']

        print("FAST_TEXT : start to train fasttext model")
        self.model = FastText(size=self.conf.vector_size,
                              window=self.conf.emb_window,
                              workers=self.conf.emb_workers,
                              min_count=self.conf.emb_min_count,
                              iter=self.conf.emb_iter)

        print('FAST_TEXT : start to build vocab')
        self.model.build_vocab(dataset)

        print('FAST_TEXT : start to train model')
        self.model.train(dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs,
                         callbacks=[EmbedCallback()])

        self.store_model()
        return self.model

    def embed(self, tokenized_text):
        if self.model is None:
            self.model = self.load_model()

        sentence_vector = []
        for word in tokenized_text:
            word_vector = self.model.wv[word]
            word_vector = torch.tensor(word_vector)
            word_vector = torch.unsqueeze(word_vector, dim=0)
            sentence_vector.append(word_vector)

        sentence_vector = torch.cat(sentence_vector, dim=0)
        return sentence_vector

    def similar_word(self, word, n=10):
        if self.model is None:
            self.model = self.load_model()

        similar_words = self.model.similar_by_word(word, topn=n)
        print(word, "와 비슷한 단어 : ")
        print([word for word, cos in similar_words])
        return similar_words

    def load_model(self):
        if self.model is None:
            self.model = FastText.load(self.conf.embed_storefile)
        return self.model

    def store_model(self):
        if not os.path.exists(self.conf.embed_storepath):
            os.makedirs(self.conf.embed_storepath)
        self.model.save(self.conf.embed_storefile)
