"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from embed.embed_processor import EmbedProcessor
from intent.intent_classifier import IntentClassifier
from intent.intent_trainer import IntentTrainer
from intent.model import intent_net

# trainers
embed = EmbedProcessor()
intent_trainer = IntentTrainer(embed, model=intent_net)
intent_trainer.train()
intent_trainer.test()
# while True: print(IntentClassifier(embed, intent_net).classify(input()))
