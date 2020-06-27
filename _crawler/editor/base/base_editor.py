import re

from _crawler.crawler.base.base_crawler import BaseCrawler
from _crawler.decorators import editor


@editor
class BaseEditor(BaseCrawler):

    def join_dict(self, _dict: dict, key: str, insert: str = "") -> dict:
        """
        딕셔너리의 value 값 리스트에 담긴 string들을 join합니다.

        before :
        dict = {
            'key' : ['val1', 'val2', 'val3']
        }

        after :
        dict = {
            'key' : 'val1 val2 val3' + insert
        }

        :param _dict: 딕셔너리
        :param key: 키값
        :param insert: 추가로 삽입할 말
        :return: 수정된 딕셔너리
        """

        if _dict[key] is not None and len(_dict[key]) != 0:
            if _dict[key][0] is not None:
                _dict[key] = ' '.join(_dict[key])
                _dict[key] += insert
        return _dict

    def enumerate_josa(self, j1: str, j2: str, list_: list) -> list:
        """
        단어들을 나열할 때, 은/는/이/가/에/에서 등의 조사만 계속 연결하면 매우 어색합니다.
        때문에 만약 리스트의 두 원소가 연속적으로 같으면 '도/에도'와 같은 조사를 추가할 수 있게 합니다.

        :param j1: 은/는/이/가/에/에서 등의 일반조사
        :param j2: 도/에도/에서도 등의 보조사
        :param list_: 단어/문장 배열
        :return: [는, 는, 도, 는, 도]와 같은 조사 배열
        """

        josa = [j1]
        for i in range(len(list_) - 1):
            if list_[i] is not None and \
                    list_[i + 1] is not None:

                if list_[i] == list_[i + 1]:
                    josa.append(j2)
                else:
                    josa.append(j1)

        return josa
