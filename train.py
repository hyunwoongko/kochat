"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from config import Config
from embed.embed_processor import EmbedProcessor
from intent.intent_classifier import IntentClassifier
from intent.model import text_cnn

conf = Config()
embed = EmbedProcessor()
intent = IntentClassifier(embed, model=text_cnn)

if __name__ == '__main__':
    embed.train()
    intent.train()
