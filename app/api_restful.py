from flask import Flask, request

from embed.embed_processor import EmbedProcessor
from entity import entity_model
from entity.entity_recognizer import EntityRecognizer
from intent.model import intent_model
from intent.classifier.intent_classifier import IntentClassifier
from util.tokenizer import Tokenizer

app = Flask(__name__)
embed = EmbedProcessor()

tokenizer = Tokenizer()
intent_classifier = IntentClassifier(embed, model=intent_model)
entity_recognizer = EntityRecognizer(embed, model=entity_model)


@app.route('/')
def init():
    return 'Chat Server On'


@app.route('/tokenize')
def tokenize():
    input = request.args.get('input', None)
    output = tokenizer.tokenize(input)
    return output


@app.route('/intent')
def intent():
    input = request.args.get('input', None)
    output = intent_classifier.classify(input)
    return output


@app.route('/entity')
def entity():
    input = request.args.get('input', None)
    output = entity_recognizer.recognize(input)
    return output


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9893)
