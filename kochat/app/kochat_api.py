from typing import List

from flask import Flask
from kochat.data.dataset import Dataset
from kochat.proc.base_processor import BaseProcessor

from app.scenario import Scenario
from app.scenario_manager import ScenarioManager


class KochatApi:

    def __init__(self,
                 dataset: Dataset,
                 embed_processor: BaseProcessor,
                 intent_classifier: BaseProcessor,
                 entity_recognizer: BaseProcessor,
                 scenarios: List[Scenario]):
        """
        Flask를 이용해 구현한 RESTFul API 클래스입니다.

        :param dataset: 데이터셋 객체
        :param embed_processor: 임베딩 프로세서 객체
        :param intent_classifier: 인텐트 분류기 객체
        :param entity_recognizer: 개체명 인식기 객체
        :param scenarios : 시나리오 리스트 (list 타입)
        """

        self.__app = Flask(__name__)
        self.__app.config['JSON_AS_ASCII'] = False
        self.dataset = dataset
        self.embed_processor = embed_processor
        self.intent_classifier = intent_classifier
        self.entity_recognizer = entity_recognizer
        self.scenario_manager = ScenarioManager()

        for scenario in scenarios:
            self.scenario_manager.add_scenario(scenario)

    def __build(self, app):
        """
        flask 함수들을 build합니다.

        :param app: Flask application
        """

        @app.route('/')
        def index():
            """
            :return: 서버 index 페이지
            """

            return "KOCHAT BUILD SUCCESS"

        @app.route('/request/<text>', methods=['GET'])
        def request(text: str) -> dict:
            """
            문자열을 입력하면 intent, entity, state, answer 등을 포함한
            딕셔너리를 json 형태로 반환합니다.
            
            :param text: 유저 입력 문자열
            :return: json 딕셔너리
            """

            prep = self.dataset.load_predict(text, self.embed_processor)
            intent = self.intent_classifier.predict(prep, calibrate=False)
            entity = self.entity_recognizer.predict(prep)
            return self.scenario_manager.apply_scenario(intent, entity, text)

        @app.route('/fill_slot/<intent>/<text>', methods=['GET'])
        def fill_slot(intent: str, text: str) -> dict:
            """
            이전 대화에서 entity가 충분히 입력되지 않았을 때
            빈 슬롯을 채우기 위해 entity recognition을 요청합니다.

            :param intent: 이전 대화의 인텐트 (발화 의도)
            :param text: 유저 입력 문자열
            :return: json 딕셔너리
            """

            prep = self.dataset.load_predict(text, self.embed_processor)
            entity = self.entity_recognizer.predict(prep)
            return self.scenario_manager.apply_scenario(text, intent, entity)

    def run(self, port: int, ip: str = '0.0.0.0'):
        """
        Kochat 서버를 실행합니다.

        :param port: 서버를 열 포트번호
        :param ip: 아이피 주소 (default : 0.0.0.0)
        """

        self.__build(app=self.__app)
        self.__app.run(host=ip, port=port)

    def fit_intent(self):
        """
        Intent Classifier를 학습합니다.

        :return: 학습된 IntentClassifier
        """

        self.intent_classifier.fit(self.dataset.load_intent(self.embed_processor))
        return self.intent_classifier

    def fit_entity(self):
        """
        Entity Recognizr를 학습합니다.

        :return: 학습된 Entity Recognizr
        """

        self.entity_recognizer.fit(self.dataset.load_entity(self.embed_processor))
        return self.entity_recognizer

    def fit_embed(self):
        """
        Embedding을 학습합니다.

        :return: 학습된 Embedding
        """

        self.embed_processor.fit(self.dataset.load_embed())
        return self.embed_processor
