"""
@auther Hyunwoong
@since {6/20/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.answerer.base.answerer import Answerer


class RestaurantAnswerer(Answerer):

    def recommendation_form(self, location, restaurant, result):
        msg = self.restaurant.format(location=location, restaurant=restaurant)
        msg += '{location} 근처의 '

        msg = self.add_msg_from_dict(result, 'category', msg, '{category}인')
        msg = self.add_msg_from_dict(result, 'name', msg, '{name}에 가보는 건 어떨까요?')
        msg = self.add_msg_from_dict(result, 'time', msg, '운영 시간은 {time}이고')
        msg = self.add_msg_from_dict(result, 'address', msg, '주소는 {address},')
        msg = self.add_msg_from_dict(result, 'phone_number', msg, '전화번호는 {phone_number}')
        msg += '입니다. 꼭 가 보시는 걸 추천드립니다. 제가 강추해요!'
        msg = msg.format(location=location, category=result['category'], name=result['name'],
                         time=result['time'], address=result['address'], phone_number=result['phone_number'])

        return msg
