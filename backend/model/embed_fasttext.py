import torch
from gensim.models import FastText

from backend.decorators import gensim, model


@gensim
@model
class EmbedFastText(FastText):

    def __init__(self):
        super().__init__(size=self.vector_size,
                         window=self.window_size,
                         workers=self.workers,
                         min_count=self.min_count,
                         iter=self.iter)

    def __call__(self, sequence):
        sentence_vector = []
        for word in sequence:
            word_vector = self.wv[word]
            word_vector = torch.tensor(word_vector)
            word_vector = torch.unsqueeze(word_vector, dim=0)
            sentence_vector.append(word_vector)

        return torch.cat(sentence_vector, dim=0)
