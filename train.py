"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from embed.embed_processor import EmbedProcessor
from intent.retrieval.intent_retrieval_trainer import IntentRetrievalTrainer
from intent.model import intent_model

# embedding train
embed = EmbedProcessor()
# embed.train()

# intent train
intent_trainer = IntentRetrievalTrainer(embed, model=intent_model)
# intent_trainer.train()
intent_trainer.test_retrieval()

# entity train
# entity_trainer = EntityTrainer(embed, model=entity_model)
# entity_trainer.train()
