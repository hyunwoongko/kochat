"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
from kochat.app import KochatApi
from kochat.data import Dataset
from kochat.model import intent
from kochat.proc import DistanceClassifier

dataset = Dataset(ood=True)

intent_classifier = DistanceClassifier(
    model=intent.CNN()
)

kochat = KochatApi()
