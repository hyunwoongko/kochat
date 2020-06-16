import torch
from gensim.models import FastText

from backend.decorators import model


@model
class Model(FastText):

    def __init__(self, vector_size, window_size, workers, min_count, iter):
        super().__init__(size=vector_size,
                         window=window_size,
                         workers=workers,
                         min_count=min_count,
                         iter=iter)

    def __call__(self, sequence):
        sentence_vector = []
        for word in sequence:
            word_vector = self.wv[word]
            word_vector = torch.tensor(word_vector)
            word_vector = torch.unsqueeze(word_vector, dim=0)
            sentence_vector.append(word_vector)

        return torch.cat(sentence_vector, dim=0)
