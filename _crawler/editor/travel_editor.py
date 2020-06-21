"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.editor.base.editor import Editor
import re

class TravelEditor(Editor):

    def edit_travel(self, location, travel, result):
        result = self.join_dict(result, 'name')
        result = self.join_dict(result, 'context')
        result = self.join_dict(result, 'category')
        result = self.join_dict(result, 'address')
        result = self.join_dict(result, 'thumUrl')

        if isinstance(result['context'], str):
            result['context'] = re.sub(' ', ', ', result['context'])

        return result
