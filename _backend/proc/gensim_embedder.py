"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from time import time

from gensim.models.callbacks import CallbackAny2Vec

from _backend.decorators import gensim
from _backend.proc.base.base_processor import BaseProcessor


@gensim
class GensimEmbedder(BaseProcessor):

    def __init__(self, model):
        super().__init__(model)

    def predict(self, sequence):
        self._load_model()
        return self.model(sequence)

    def fit(self, dataset):
        self.model.build_vocab(dataset)
        self.model.train(sentences=dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs,
                         callbacks=[self.Logger(self.__class__.__name__)])

        self._save_model()
        return self.model

    def test(self):
        raise Exception("임베딩은 테스트 할 수 없습니다.")

    def _load_model(self):
        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.model_loaded = True
            self.model = self.model.__class__.load(self.model_file + '.gensim')

    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model.save(self.model_file + '.gensim')

    class Logger(CallbackAny2Vec):

        def __init__(self, name):
            self.epoch = 0
            self.eta = 0
            self.name = name

        def on_epoch_begin(self, model):
            self.eta = time()

        def on_epoch_end(self, model):
            print('{name} - epoch: {epoch}, eta: {sec}(sec)'
                  .format(name=self.name,
                          epoch=self.epoch,
                          sec=round(time() - self.eta, 4)))

            self.epoch += 1
            self.eta = 0
