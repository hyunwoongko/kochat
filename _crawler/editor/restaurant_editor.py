"""
@auther Hyunwoong
@since {6/20/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.editor.base.base_editor import BaseEditor


class RestaurantEditor(BaseEditor):

    def edit_restaurant(self, result: dict) -> dict:
        """
        join_dict를 사용하여 딕셔너리에 있는 string 배열들을
        하나의 string으로 join합니다.

        :param location: 지역
        :param travel: 여행지
        :param result: 데이터 딕셔너리
        :return: 수정된 딕셔너리
        """

        result = self.join_dict(result, 'name')
        result = self.join_dict(result, 'phone_number')
        result = self.join_dict(result, 'category', insert="전문점")
        result = self.join_dict(result, 'time')
        result = self.join_dict(result, 'address')
        return result
