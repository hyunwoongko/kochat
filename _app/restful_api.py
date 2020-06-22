from flask import Flask

from _app.scenario_manager import ScenarioManager


class KochatApi:

    def __init__(self, dataset, embed_processor, intent_classifier, entity_recognizer):
        self.__app = Flask(__name__)
        self.__app.config['JSON_AS_ASCII'] = False
        self.scenario = ScenarioManager()
        self.dataset = dataset
        self.embed_processor = embed_processor
        self.intent_classifier = intent_classifier
        self.entity_recognizer = entity_recognizer

    def build(self, app):
        @app.route('/')
        def index():
            return "KOCHAT BUILD SUCCESS"

        @app.route('/request/<text>', methods=['GET'])
        def request(text):
            prep = self.dataset.load_predict(text, self.embed_processor)
            intent = self.intent_classifier.predict(prep, calibrate=False)
            entity = self.entity_recognizer.predict(prep)
            return self.scenario.apply_scenario(text, intent, entity)

        @app.route('/fill_info/<intent>/<text>', methods=['GET'])
        def fill_info(intent, text):
            prep = self.dataset.load_predict(text, self.embed_processor)
            entity = self.entity_recognizer.predict(prep)
            return self.scenario.apply_scenario(text, intent, entity)

    def run(self, ip, port):
        self.build(app=self.__app)
        self.__app.run(host=ip, port=port)
