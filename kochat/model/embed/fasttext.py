import torch
from gensim.models import FastText
from torch import Tensor
from kochat.decorators import gensim


@gensim
class FastText(FastText):

    def __init__(self):
        """
        Gensim Fasttext 모델의 Wrapper 클래스입니다.
        """

        super().__init__(size=self.vector_size,
                         window=self.window_size,
                         workers=self.workers,
                         min_count=self.min_count,
                         iter=self.iter)

    def __call__(self, sequence: str):
        return self.forward(sequence)

    def forward(self, sequence: str) -> Tensor:
        sentence_vector = []

        for word in sequence:
            word_vector = self.wv[word]  # 단어 → 벡터
            word_vector = torch.tensor(word_vector)  # 벡터 → 토치텐서
            word_vector = torch.unsqueeze(word_vector, dim=0)  # concat을 위해 unsqueeze
            sentence_vector.append(word_vector)

        return torch.cat(sentence_vector, dim=0)  # concatenation
