"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""

from gensim.models import FastText
from kochat.decorators import gensim


@gensim
class FastText(FastText):

    def __init__(self):
        """
        Gensim Word2Vec 모델의 Wrapper 클래스입니다.
        """

        super().__init__(size=self.vector_size,
                         window=self.window_size,
                         workers=self.workers,
                         min_count=self.min_count,
                         iter=self.iter)
