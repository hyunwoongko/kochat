"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from time import time

from gensim.models.callbacks import CallbackAny2Vec

from backend.decorators import gensim
from backend.proc.base.base_processor import BaseProcessor
from util.oop import override


@gensim
class GensimEmbedder(BaseProcessor):

    def __init__(self, model):
        super().__init__(model)

    @override(BaseProcessor)
    def train(self, dataset):
        self.model.build_vocab(dataset)
        self.model.train(sentences=dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs,
                         callbacks=[self.Callback(self.model.name)])

        self._save_model()
        return self.model

    @override(BaseProcessor)
    def test(self):
        raise Exception("임베딩은 테스트 할 수 없습니다.")

    @override(BaseProcessor)
    def inference(self, sequence):
        return self.model(sequence)

    @override(BaseProcessor)
    def _load_model(self):
        if not self.model_loaded:
            self.model_loaded = True
            self.model = self.model.load(self.model_file + '.gensim')

    @override(BaseProcessor)
    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model.save(self.model_file + '.gensim')

    class Callback(CallbackAny2Vec):

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
