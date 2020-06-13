from flask import Flask, request

from embed.embed_processor import EmbedProcessor
from intent.intent_classifier import IntentClassifier
from intent import intent_model

app = Flask(__name__)
embed = EmbedProcessor()
intent_classifier = IntentClassifier(embed, model=intent_model)


@app.route('/')
def init():
    return 'Chat Server On'


@app.route('/intent')
def intent():
    user_input = request.args.get('intent', None)
    model_output = intent_classifier.classify(user_input)
    return {"input": user_input, "output": model_output}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9893)
