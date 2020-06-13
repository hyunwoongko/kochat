from flask import Flask, request

from embed.embed_processor import EmbedProcessor
from intent.intent_classifier import IntentClassifier
from intent import intent_model
from util.tokenizer import Tokenizer

app = Flask(__name__)
embed = EmbedProcessor()

tokenizer = Tokenizer()
intent_classifier = IntentClassifier(embed, model=intent_model)


@app.route('/')
def init():
    return 'Chat Server On'


@app.route('/tokenize')
def intent():
    input = request.args.get('input', None)
    output = tokenizer.tokenize(input)
    return {"input": input, "output": output}


@app.route('/intent')
def intent():
    input = request.args.get('input', None)
    output = intent_classifier.classify(input)
    return {"input": input, "output": output}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9893)
