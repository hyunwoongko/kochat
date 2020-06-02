"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from embed.embed_processor import EmbedProcessor
from intent.intent_classifier import IntentClassifier
from intent.intent_trainer import IntentTrainer
from intent.siamese_trainer import SiameseTrainer
from intent.model import \
    resnet5, \
    resnet9, \
    lstm_uni, \
    lstm_bi

# trainers
embed = EmbedProcessor()
intent_trainer = IntentTrainer(embed, model=resnet5)
# intent_trainer = SiameseClassifier(embed, model=resnet5)

# inferences
intent_classifier = IntentClassifier(embed)

if __name__ == '__main__':
    # embed.train()
    # intent_trainer.train()
    # intent_trainer.test()
    intent_classifier.inference("오늘 전주 날씨 어때")
