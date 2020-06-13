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
    """
    단어, 문장을 텐서로 임베딩하는 클래스입니다.
    임베딩 방법은 fasttext이며, 추후에 ELMO등을 추가할 예정입니다.

    TODO : ELMO 임베딩, GLOVE 임베딩 등 임베딩 클래스 추가하기
    """

    tok = Tokenizer()
    conf = Config()
    data = Dataset()
    model = None

    def train(self):
        """
        데이터셋을 로드하여 임베딩 모델을 학습합니다.
        :return: 모델을 저장하고 리턴합니다.
        """
        print('FAST_TEXT : start to make dataset')
        dataset = self.data.embed_train()['data']

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
        """
        TOKENIZING 된 문장을 입력받아서 2차원(Matrix)로 변환합니다.
        ['안녕', '나는', '개발자야'] => [3, vector_size]

        :param tokenized_text: 토크나이징 된 단어 열이 입력되어야 합니다.
        :return: 임베딩된 Matrix를 리턴합니다.
        """
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
        """
        단어를 입력하면 유사한 단어들의 목록을 보여줍니다.

        :param word: 유사도를 확인할 단어입니다.
        :param n: 몇개를 보여줄지 설정합니다 (default=10)
        :return: 유사한 단어 리스트를 리턴합니다.
        """

        if self.model is None:
            self.model = self.load_model()

        similar_words = self.model.similar_by_word(word, topn=n)
        print(word, "와 비슷한 단어 : ")
        print([word for word, cos in similar_words])
        return similar_words

    def load_model(self):
        """
        모델이 None이라면 저장된 Weight를 불러와서 모델에 로드합니다.

        :return: 로드된 모델
        """

        if self.model is None:
            self.model = FastText.load(self.conf.embed_storefile)

        return self.model

    def store_model(self):
        """
        학습이 종료되면 모델의 Weight를 지정한 공간에 저장합니다.
        """

        if not os.path.exists(self.conf.embed_storepath):
            os.makedirs(self.conf.embed_storepath)

        self.model.save(self.conf.embed_storefile)
