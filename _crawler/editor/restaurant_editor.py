"""
@auther Hyunwoong
@since {6/20/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.editor.base.editor import Editor


class RestaurantEditor(Editor):

    def edit_restaurant(self, result: dict):
        result = self.join_dict(result, 'name')
        result = self.join_dict(result, 'phone_number')
        result = self.join_dict(result, 'category', insert="전문점")
        result = self.join_dict(result, 'time')
        result = self.join_dict(result, 'address')
        return result
