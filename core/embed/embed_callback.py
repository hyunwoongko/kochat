"""
@author : Hyunwoong
@when : 5/10/2020
@homepage : https://github.com/gusdnd852
"""
from time import time
from gensim.models.callbacks import CallbackAny2Vec

class EmbedCallback(CallbackAny2Vec):
    """
    GENSIM 임베딩 학습시 로그를 출력합니다.
    fasttext의 경우 Loss는 표시되지 않고 ETA만 표시 가능합니다.
    """

    def __init__(self):
        self.epoch = 0
        self.eta = 0

    def on_epoch_begin(self, model):
        self.eta = time()

    def on_epoch_end(self, model):
        print('EMBEDDING : epoch {} ({}sec.)'.format(self.epoch, round(time() - self.eta, 4)))
        self.epoch += 1
        self.eta = 0
