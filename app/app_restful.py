from flask import Flask, request

from util.tokenizer import Tokenizer
from core.embed.embed_processor import EmbedProcessor
from core.intent.classifier import IntentClassifier
from core.intent import intent_model
from core.entity import EntityRecognizer, entity_model


class RestfulApi:
    app = Flask(__name__)
    tokenizer = Tokenizer()
    embed = EmbedProcessor()

    intent_classifier = IntentClassifier(embed, model=intent_model)
    entity_recognizer = EntityRecognizer(embed, model=entity_model)

    def tokenize(self):
        input = request.args.get('input', None)
        output = self.tokenizer.tokenize(input)
        return output

    def intent(self):
        input = request.args.get('input', None)
        output = self.intent_classifier.classify(input)
        return output

    def entity(self):
        input = request.args.get('input', None)
        output = self.entity_recognizer.recognize(input)
        return output
