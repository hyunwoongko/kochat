"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.base_scenario import BaseScenario


class FallbackScenario(BaseScenario):

    def __call__(self):
        return {'intent': 'FALLBACK',
                'entity': 'FALLBACK',
                'answer': 'FALLBACK',
                'state': 'FALLBACK'}
