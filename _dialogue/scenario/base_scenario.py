"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""


class BaseScenario:

    def _check_entity(self, text: list, entity: list, dict_: dict) -> dict:
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

    def _set_as_default(self, list_: list, default: str) -> list:
        """
        해당 엔티티의 길이가 0이면 디폴트 값을 추가합니다.

        :param list_: 엔티티 리스트
        :param default: 디폴트 엔티티
        :return: 디폴트 값이 추가된 리스트
       """

        if len(list_) == 0:
            list_.append(default)

        return list_
