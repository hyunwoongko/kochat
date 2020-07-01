"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
import collections
from collections import Callable
from random import randint


class Scenario:

    def __init__(self, intent, api, scenario_dict):
        self.intent = intent
        self.scenario_dict, self.default_dict = \
            self.__make_empty_dict(scenario_dict)

        self.api, self.dict_keys, self.params = \
            self.__check_api(api)

    def __check_api(self, api):
        if not isinstance(api, Callable):
            raise Exception('\n\n'
                            '0반드시 api는 callable 해야합니다.\n'
                            '입력하신 api의 타입은 {}입니다.\n'
                            '가급적이면 함수 이름 자체를 입력해주세요.'.format(type(api)))

        parameters = list(api.__code__.co_varnames)
        if 'self' in parameters: del parameters[0]
        # 만약 클래스의 멤버라면 self 인자를 지웁니다.
        dict_keys = list(self.scenario_dict.keys())

        if len(parameters) != len(dict_keys):
            raise Exception('\n\n'
                            '엔티티의 종류와 입력하신 API의 파라미터 수가 맞지 않습니다.\n'
                            '시나리오에 정의된 엔티티의 종류와 API의 파라미터 수는 일치해야합니다.\n'
                            '- 시나리오 엔티티 : {0}, {1}개\n'
                            '- API의 파라미터 : {2}, {3}개'.format(dict_keys, len(dict_keys),
                                                             parameters, len(parameters)))

        for entity in zip(parameters, dict_keys):
            api_param = entity[0]
            dict_key = entity[1]

            if api_param.lower() != dict_key.lower():
                raise Exception('\n\n'
                                'Kochat은 최대한 정확한 기능 수행을 위해 API의 파라미터의 이름과 순서를 고려하여 엔티티와 맵핑합니다.\n'
                                'API 파라미터 이름과 시나리오의 엔티티의 \'순서\'와 \'이름\'을 가급적이면 동일하게 맞춰주시길 바랍니다.\n'
                                'API 파라미터 이름과 시나리오의 엔티티는 철자만 동일하면 됩니다, 대/소문자는 일치시킬 필요 없습니다.\n'
                                '- 시나리오 엔티티 : {0}\n'
                                '- API의 파라미터 : {1}\n'
                                '- 일치하지 않은 부분 : {2} != {3}'.format(dict_keys, parameters,
                                                                   api_param, dict_key))

        return api, dict_keys, parameters

    def __make_empty_dict(self, scenario_dict):
        default_dict = {}

        for k, v in scenario_dict.items():
            if len(scenario_dict[k]) > 0:
                default_dict[k] = v
                # 디폴트 딕셔너리로 일단 빼놓고

                scenario_dict[k] = []
                # 해당 엔티티의 리스트를 비워둠

            else:
                default_dict[k] = []
                # 디폴드 없으면 리스트로 초기화

        return scenario_dict, default_dict

    def __check_entity(self, entity: list, text: list, dict_: dict) -> dict:
        """
        문자열과 엔티티를 함께 체크해서
        딕셔너리에 정의된 엔티티에 해당하는 단어 토큰만 추가합니다.

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :param dict_: 필요한 엔티티가 무엇인지 정의된 딕셔너리
        :return: 필요한 토큰들이 채워진 딕셔너리
        """

        for t, e in zip(text, entity):
            for k, v in dict_.items():
                if k in e:
                    v.append(t)

        return dict_

    def __set_default(self, result_dict):
        for k, v in result_dict.items():
            if len(result_dict[k]) == 0 and len(self.default_dict[k]) != 0:
                # 디폴트 값 중에서 랜덤으로 하나 골라서 넣음
                result_dict[k] = \
                    [self.default_dict[k][randint(0, len(self.default_dict[k]) - 1)]]

            result_dict[k] = ' '.join(result_dict[k])
        return result_dict

    def apply(self, entity, text):
        result_dict = self.__check_entity(entity, text, self.scenario_dict)
        result_dict = self.__set_default(result_dict)
        required_entity = [k for k, v in result_dict.items() if len(v) == 0]

        if len(required_entity) == 0:
            return {
                'input': text,
                'intent': self.intent,
                'entity': entity,
                'state': 'SUCCESS',
                'answer': self.api(*result_dict.values())
            }

        else:
            return {
                'input': text,
                'intent': self.intent,
                'entity': entity,
                'state': 'REQUIRE_' + '_'.join(required_entity),
                'answer': None
            }
