"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from embed.embed_processor import EmbedProcessor
from intent.intent_classifier import IntentClassifier
from intent.model import text_cnn

if __name__ == '__main__':
    embed = EmbedProcessor()
    embed.train()
    # intent = IntentClassifier(embed, model=text_cnn)
    # intent.train()