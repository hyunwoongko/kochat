"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from embed.embed_processor import EmbedProcessor
from entity import entity_model
from entity.entity_trainer import EntityTrainer
from intent.intent_classifier import IntentClassifier
from intent.intent_trainer import IntentTrainer
from intent import intent_model

# embedding train
embed = EmbedProcessor()
# embed.train()

# intent train
# intent_trainer = IntentTrainer(embed, model=intent_model)
# intent_trainer.train()
# intent_trainer.test()
# while True: print(IntentClassifier(embed, intent_model).classify(input()))

# entity train
entity_trainer = EntityTrainer(embed, model=entity_model)
entity_trainer.train()
entity_trainer.test()
