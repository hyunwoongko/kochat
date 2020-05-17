"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
import torch
from gensim.models import FastText

from configs import GlobalConfigs
from embed.embed_callback import EmbedCallback
from embed.embed_visualizer import EmbedVisualizer
from util.dataset import TrainDataLoader
from util.tokenizer import Tokenizer


class EmbedProcessor:
    tok = Tokenizer()
    conf = GlobalConfigs()
    model = None
    dataset = None

    def __init__(self, data_path, store_path):
        self.data_path = data_path
        self.store_path = store_path

    def train(self):
        print("FAST_TEXT : start to train fasttext model")
        self.dataset = self.make_dataset(self.data_path)

        self.model = FastText(size=self.conf.vector_size,
                              window=self.conf.emb_window,
                              workers=self.conf.emb_workers,
                              min_count=self.conf.emb_min_count,
                              iter=self.conf.emb_iter)

        print('FAST_TEXT : start to build vocab')
        self.model.build_vocab(self.dataset)

        print('FAST_TEXT : start to train model')
        self.model.train(self.dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs,
                         callbacks=[EmbedCallback()])

        self.store_model()
        self.visualize()
        return self.model

    def make_dataset(self, data_path):
        print('FAST_TEXT : start to make dataset')
        loader = TrainDataLoader()
        intent = loader.load_intent(data_path)
        dataset = intent['data']
        return dataset

    def embed(self, text):
        if self.model is None:
            self.model = self.load_model()

        sentence_vector = []
        for word in text:
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
            self.model = FastText.load(self.store_path + "\\fasttext.model")

        return self.model

    def store_model(self):
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

        self.model.save(self.store_path + "\\fasttext.model")

    def visualize(self):
        emb_vis = EmbedVisualizer(self.conf)
        print("FAST_TEXT : start t-sne visualization")
        emb_vis.visualize(self.model)
