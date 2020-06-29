"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os
from time import time

from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.callbacks import CallbackAny2Vec
from torch import Tensor

from kochat.decorators import gensim
from kochat.proc.base.base_processor import BaseProcessor


@gensim
class GensimEmbedder(BaseProcessor):

    def __init__(self, model: BaseWordEmbeddingsModel):
        """
        Gensim 모델의 Training, Inference
        등을 관장하는 프로세서 클래스입니다.

        :param model: Gensim 모델을 입력해야합니다.
        """

        super().__init__(model)
        self.callback = self.GensimLogger(
            name=self.__class__.__name__,
            logging=self._print
        )  # 학습 진행사항 출력 콜백

    def fit(self, dataset: list):
        """
        데이터셋으로 Vocabulary를 생성하고
        모델을 학습 및 저장시킵니다.

        :param dataset: 데이터셋
        :return: 학습된 모델을 리턴합니다.
        """

        self.model.build_vocab(dataset)
        self.model.train(
            sentences=dataset,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs + 1,
            callbacks=[self.callback]
        )

        self._save_model()

    def predict(self, sequence: str) -> Tensor:
        """
        사용자의 입력을 임베딩합니다.

        :param sequence: 입력 시퀀스
        :return: 임베딩 벡터 반환
        """

        self._load_model()
        return self.model(sequence)

    def _load_model(self):
        """
        저장된 모델을 불러옵니다.
        """

        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.model_loaded = True
            self.model = self.model.__class__.load(self.model_file + '.gensim')

    def _save_model(self):
        """
        모델을 저장장치에 저장합니다.
        """

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model.save(self.model_file + '.gensim')

    class GensimLogger(CallbackAny2Vec):

        def __init__(self, name: str, logging):
            """
            Gensim 모델의 학습 과정을 디버깅하기 위한 callback

            :param name: 모델 이름
            :param print: base processor의 print 함수를 전달받습니다.
            """

            self.epoch, self.eta = 0, 0
            self.name = name
            self.logging = logging

        def on_epoch_begin(self, model: BaseWordEmbeddingsModel):
            """
            epoch 시작시에 시간 측정을 시작합니다.

            :param model: 학습할 모델
            """
            self.eta = time()

        def on_epoch_end(self, model: BaseWordEmbeddingsModel):
            """
            epoch 종료시에 걸린 시간을 체크하여 출력합니다.

            :param model: 학습할 모델
            """

            self.logging(
                name=self.name,
                msg='Epoch : {epoch}, ETA : {sec} sec'
                    .format(epoch=self.epoch, sec=round(time() - self.eta, 4))
            )

            self.epoch += 1
