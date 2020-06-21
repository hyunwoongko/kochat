"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.answerer.base.answerer import Answerer


class TravelAnswerer(Answerer):

    def travel_form(self, location, travel, result):
        msg = self.travel.format(location=location)
        msg += '{location} 근처의 '

        msg = self.add_msg_from_dict(result, 'context', msg, '{context}를 즐길 수 있는')
        msg = self.add_msg_from_dict(result, 'category', msg, '{category}')
        msg = self.add_msg_from_dict(result, 'name', msg, '{name}에 가보시는 건 어떤가요?')
        msg = self.add_msg_from_dict(result, 'address', msg, '주소는 {address}입니다.')
        msg = self.add_msg_from_dict(result, 'thumUrl', msg, '> 사진보기 : {thumUrl}')
        msg = msg.format(location=location, context=result['context'], category=result['category'],
                         name=result['name'], address=result['address'], thumUrl=result['thumUrl'])

        return msg
