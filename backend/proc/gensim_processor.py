"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from time import time

from gensim.models import FastText

from backend.decorators import gensim
from backend.proc.base_processor import BaseProcessor
from util.oop import override
from gensim.models.callbacks import CallbackAny2Vec


@gensim
class GensimProcessor(BaseProcessor):

    def _load_model(self):
        pass

    def __init__(self, model):
        super().__init__(model.Model(vector_size=self.vector_size,
                                     window_size=self.window_size,
                                     workers=self.workers,
                                     min_count=self.min_count,
                                     iter=self.iter))

    @override(BaseProcessor)
    def train(self, dataset):
        dataset, _ = dataset
        self.model.build_vocab(dataset)
        self.model.train(sentences=dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs,
                         callbacks=[self.Callback(self.model.name)])

        self._save_model()
        return self.model

    @override(BaseProcessor)
    def inference(self, sequence):
        return self.model(sequence)

    # @override(BaseProcessor)
    def load_model(self):
        self.model = FastText.load(self.model_file)
        self.model_loaded = True

    @override(BaseProcessor)
    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model.save(self.model_file)

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
