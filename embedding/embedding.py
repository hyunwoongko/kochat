"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
import torch
from gensim.models import FastText

from configs import FastTextConfigs, GlobalConfigs
from embedding.callback import EmbeddingCallback
from util.loader import TrainDataLoader
from util.tokenizer import Tokenizer


class Embedding:
    tok = Tokenizer()
    conf = FastTextConfigs()
    glb_conf = GlobalConfigs()
    model = None
    dataset = None

    def __init__(self, store_path):
        self.store_path = store_path

    def train(self, data_path):
        print("FAST_TEXT : start to train fasttext model")
        self.dataset = self.make_dataset(data_path)

        self.model = FastText(size=self.glb_conf.vector_size,
                              window=self.conf.window,
                              workers=self.conf.workers,
                              min_count=self.conf.min_count,
                              iter=self.conf.iter)

        print('FAST_TEXT : start to build vocab')
        self.model.build_vocab(self.dataset)

        print('FAST_TEXT : start to train model')
        self.model.train(self.dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs,
                         callbacks=[EmbeddingCallback()])

        self.store_model()
        print("FAST_TEXT : finish training fasttext model.")
        return self.model

    def make_dataset(self, data_path):
        print('FAST_TEXT : start to make dataset')
        loader = TrainDataLoader()
        intent = loader.load_intent(data_path)
        dataset = intent['data']
        return dataset

    def store_model(self):
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

        self.model.save(self.store_path + "\\fasttext.model")

    def embed_multiple_rows(self, texts):
        document_vector = []
        for text in texts:
            sentence_vector = self.embed_single_row(text)
            sentence_vector = torch.unsqueeze(sentence_vector, dim=0)
            document_vector.append(sentence_vector)

        document_vector = torch.cat(document_vector, dim=0)
        return document_vector

    def embed_single_row(self, text):
        if self.model is None:
            self.model = FastText.load(self.store_path + "\\fasttext.model")

        sentence_vector = []
        for word in text:
            word_vector = self.model.wv[word]
            word_vector = torch.tensor(word_vector)
            word_vector = torch.unsqueeze(word_vector, dim=0)
            sentence_vector.append(word_vector)

        sentence_vector = torch.cat(sentence_vector, dim=0)
        return sentence_vector

    def similar_word(self, word, n=2):
        if self.model is None:
            self.model = FastText.load(self.store_path + "\\fasttext.model")

        similar_words = self.model.similar_by_word(word, topn=n)
        print(word, "와 비슷한 단어 : ")
        print([word for word, cos in similar_words])
        return similar_words

    def load_model(self):
        if self.model is None:
            self.model = FastText.load(self.store_path + "\\fasttext.model")

        return self.model
