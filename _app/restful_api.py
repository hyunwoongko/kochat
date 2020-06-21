from flask import Flask

from _app.scenario_manager import ScenarioManager
from _backend.data.dataset import Dataset
from _backend.data.preprocessor import Preprocessor
from _backend.loss.center_loss import CenterLoss
from _backend.loss.crf_loss import CRFLoss
from _backend.model.embed_fasttext import EmbedFastText
from _backend.model.entity_lstm import EntityLSTM
from _backend.model.intent_cnn import IntentCNN
from _backend.proc.entity_recognizer import EntityRecognizer
from _backend.proc.gensim_embedder import GensimEmbedder
from _backend.proc.intent_classifier import IntentClassifier


class KochatApi:

    def __init__(self):
        self.__app = Flask(__name__)
        self.__app.config['JSON_AS_ASCII'] = False
        self.scenario = ScenarioManager()

        self.dataset = Dataset(
            preprocessor=Preprocessor(),
            ood=True)

        self.embed_processor = GensimEmbedder(
            model=EmbedFastText())

        self.intent_classifier = IntentClassifier(
            model=IntentCNN(self.dataset.intent_dict),
            loss=CenterLoss(self.dataset.intent_dict))

        self.entity_recognizer = EntityRecognizer(
            model=EntityLSTM(self.dataset.entity_dict),
            loss=CRFLoss(self.dataset.entity_dict))

    def build(self, app):
        @app.route('/')
        def index():
            return "BUILD SUCCESS"

        @app.route('/request/<text>', methods=['GET'])
        def request(text):
            prep = self.dataset.load_predict(text, self.embed_processor)
            intent = self.intent_classifier.predict(prep, calibrate=False)
            entity = self.entity_recognizer.predict(prep)
            return self.scenario.apply_scenario(text, intent, entity)

    def run(self, ip, port):
        self.build(app=self.__app)
        self.__app.run(host=ip, port=port)
