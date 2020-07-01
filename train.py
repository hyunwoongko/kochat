"""
@auther Hyunwoong
@since 6/28/2020
@see https://github.com/gusdnd852
"""

from kochat.data import Dataset
from kochat.loss import CRFLoss
from kochat.loss import CrossEntropyLoss
from kochat.model import intent, embed, entity
from kochat.proc import DistanceClassifier
from kochat.proc import EntityRecognizer
from kochat.proc import GensimEmbedder
from kochat.proc import SoftmaxClassifier

dataset = Dataset(ood=True)

emb = GensimEmbedder(
    model=embed.FastText()
)

softmax = SoftmaxClassifier(
    model=intent.LSTM(dataset.intent_dict),
    loss=CrossEntropyLoss(dataset.intent_dict)
)

distance = DistanceClassifier(
    model=intent.LSTM(dataset.intent_dict),
    loss=CrossEntropyLoss(dataset.intent_dict)
)

entity = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

# emb.fit(dataset.load_embed())
softmax.fit(dataset.load_intent(emb))
distance.fit(dataset.load_intent(emb))
