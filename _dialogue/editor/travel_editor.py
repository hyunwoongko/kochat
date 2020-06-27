"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _dialogue.editor.base.base_editor import BaseEditor
import re


class TravelEditor(BaseEditor):

    def edit_travel(self, location: str, travel: str, result: dict) -> dict:
        """
        join_dict를 사용하여 딕셔너리에 있는 string 배열들을
        하나의 string으로 join합니다.

        :param location: 지역
        :param travel: 여행지
        :param result: 데이터 딕셔너리
        :return: 수정된 딕셔너리
        """

        result = self.join_dict(result, 'name')
        result = self.join_dict(result, 'context')
        result = self.join_dict(result, 'category')
        result = self.join_dict(result, 'address')
        result = self.join_dict(result, 'thumUrl')

        if isinstance(result['context'], str):
            result['context'] = re.sub(' ', ', ', result['context'])

        return result
