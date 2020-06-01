"""
@author : Hyunwoong
@when : 5/10/2020
@homepage : https://github.com/gusdnd852
"""
from time import time
from gensim.models.callbacks import CallbackAny2Vec


class EmbedCallback(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0
        self.eta = 0

    def on_epoch_begin(self, model):
        self.eta = time()

    def on_epoch_end(self, model):
        print('FAST_TEXT : Epoch {} was finished - EAT : {} sec.'.format(self.epoch, time() - self.eta))
        self.epoch += 1
        self.eta = 0
