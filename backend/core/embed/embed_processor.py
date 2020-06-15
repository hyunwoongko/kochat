"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os

import torch
from gensim.models import FastText

from backend.base.model_manager import Embedding
from backend.core.embed.embed_callback import EmbedCallback


class EmbedProcessor(Embedding):
    """
    단어, 문장을 텐서로 임베딩하는 클래스입니다.
    임베딩 방법은 fasttext이며, 추후에 ELMO 등을 추가할 예정입니다.

    TODO : ELMO 임베딩, GLOVE 임베딩 등 임베딩 클래스 추가하기
    """

    def __init__(self):
        super().__init__()
        self.model = None

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

    def train_model(self, dataset):
        """
        데이터셋을 로드하여 임베딩 모델을 학습합니다.
        :return: 모델을 저장하고 리턴합니다.
        """
        print("EMBEDDING : creating model")
        self.model = FastText(size=self.vector_size,
                              window=self.window_size,
                              workers=self.workers,
                              min_count=self.min_count,
                              iter=self.iter)

        print('EMBEDDING : building vocab')
        dataset, _ = dataset
        self.model.build_vocab(dataset)

        print('EMBEDDING : training model')
        self.model.train(dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs,
                         callbacks=[EmbedCallback()])

        print('EMBEDDING : saving model')
        self.store_model()
        return self.model

    def load_model(self):
        """
        모델이 None이라면 저장된 Weight를 불러와서 모델에 로드합니다.

        :return: 로드된 모델
        """

        if self.model is None:
            self.model = FastText.load(self.embed_processor_file)

        return self.model

    def store_model(self):
        """
        학습이 종료되면 모델의 Weight를 지정한 공간에 저장합니다.
        """

        if not os.path.exists(self.embed_dir):
            os.makedirs(self.embed_dir)

        self.model.save(self.embed_processor_file)
