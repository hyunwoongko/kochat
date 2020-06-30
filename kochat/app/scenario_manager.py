"""
@auther Hyunwoong
@since 6/28/2020
@see https://github.com/gusdnd852
"""
from app.scenario import Scenario


class ScenarioManager:

    def __init__(self):
        self.scenarios = []

    def add_scenario(self, scen: Scenario):
        if isinstance(scen, Scenario):
            self.scenarios.append(scen)
        else:
            raise Exception('시나리오 객체만 입력 가능합니다.')

    def apply_scenario(self, intent, entity, text):
        for scenario in self.scenarios:
            if scenario.intent == intent:
                scenario.apply(entity, text)

        return {
            'input': text,
            'intent': intent,
            'entity': entity,
            'state': 'FALLBACK',
            'answer': None
        }
