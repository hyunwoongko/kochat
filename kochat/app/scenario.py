# Copyright 2020 Kochat. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
from collections import Callable
from copy import deepcopy
from random import randint
from kochat.decorators import data


@data
class Scenario:

    def __init__(self, intent, api, scenario=None):
        self.intent = intent
        self.scenario, self.default = \
            self.__make_empty_dict(scenario)

        self.api, self.dict_keys, self.params = \
            self.__check_api(api)

    def __check_api(self, api):
        if not isinstance(api, Callable):
            raise Exception('\n\n'
                            '0반드시 api는 callable 해야합니다.\n'
                            '입력하신 api의 타입은 {}입니다.\n'
                            '가급적이면 함수 이름 자체를 입력해주세요.'.format(type(api)))

        dict_keys = list(self.scenario.keys())
        pre_defined_entity = [entity.lower() for entity in self.NER_categories]
        parameters = inspect.getfullargspec(api).args
        if 'self' in parameters: del parameters[0]
        # 만약 클래스의 멤버라면 self 인자를 지웁니다.

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

            if dict_key.lower() not in pre_defined_entity:
                raise Exception('\n\n'
                                'Kochat은 최대한 정확한 기능 수행을 위해 Config값에 정의된 Entity만 허용합니다. \n'
                                '- config에 정의된 엔티티 : {0}\n'
                                '- 시나리오 엔티티 : {1}\n'
                                '- 일치하지 않은 부분 : {2} not in {0}'.format(pre_defined_entity, dict_keys, dict_key))

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

    def __make_empty_dict(self, scenario):
        default = {}

        for k, v in scenario.items():
            if len(scenario[k]) > 0:
                default[k] = v
                # 디폴트 딕셔너리로 일단 빼놓고

                scenario[k] = []
                # 해당 엔티티의 리스트를 비워둠

            else:
                default[k] = []
                # 디폴드 없으면 리스트로 초기화

        return scenario, default

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
                if k.lower() in e.lower():
                    v.append(t)

        return dict_

    def __set_default(self, result):
        for k, v in result.items():
            if len(result[k]) == 0 and len(self.default[k]) != 0:
                # 디폴트 값 중에서 랜덤으로 하나 골라서 넣음
                result[k] = \
                    [self.default[k][randint(0, len(self.default[k]) - 1)]]

            result[k] = ' '.join(result[k])
        return result

    def apply(self, entity, text):
        scenario = deepcopy(self.scenario)
        result = self.__check_entity(entity, text, scenario)
        result = self.__set_default(result)
        required_entity = [k for k, v in result.items() if len(v) == 0]

        if len(required_entity) == 0:
            return {
                'input': text,
                'intent': self.intent,
                'entity': entity,
                'state': 'SUCCESS',
                'answer': self.api(*result.values())
            }

        else:
            return {
                'input': text,
                'intent': self.intent,
                'entity': entity,
                'state': 'REQUIRE_' + '_'.join(required_entity),
                'answer': None
            }
