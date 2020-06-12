"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from embed.embed_processor import EmbedProcessor
from intent.center_trainer import CenterTrainer
from intent.intent_classifier import IntentClassifier
from intent.intent_trainer import IntentTrainer
from intent.siamese_trainer import SiameseTrainer
from intent.model import \
    resnet5, \
    lstm_uni, \
    lstm_bi

# trainers
embed = EmbedProcessor()
intent_trainer = IntentTrainer(embed, model=resnet5)
siamese_trainer = SiameseTrainer(embed, model=resnet5)
center_trainer = CenterTrainer(embed, model=resnet5)

# classifiers
intent_classifier = IntentClassifier(embed, model=resnet5)


def train_embed():
    embed.train()


def train_intent():
    intent_trainer.train()
    intent_trainer.test()

def train_siamese():
    # siamese_trainer.train()
    siamese_trainer.test()

def train_center():
    center_trainer.train()
    center_trainer.test()

def classify_intent():
    while True:
        print("입력하세요 : ", end="")
        out = intent_classifier.classify(input())
        print(out)

if __name__ == '__main__':
    # train_embed()
    train_intent()
    # train_siamese()
    # train_center()