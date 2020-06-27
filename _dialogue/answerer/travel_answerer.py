"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _dialogue.answerer.base.base_answerer import BaseAnswerer


class TravelAnswerer(BaseAnswerer):

    def travel_form(self, location: str, travel: str, result: dict) -> str:
        """
        여행지 출력 포맷
        
        :param location: 지역
        :param travel: 여행지
        :param result: 데이터 딕셔너리
        :return: 출력 메시지
        """

        msg = self.travel.format(location=location)
        msg += '{location} 근처의 '

        msg = self._add_msg_from_dict(result, 'context', msg, '{context}를 즐길 수 있는')
        msg = self._add_msg_from_dict(result, 'category', msg, '{category}')
        msg = self._add_msg_from_dict(result, 'name', msg, '{name}에 가보시는 건 어떤가요?')
        msg = self._add_msg_from_dict(result, 'address', msg, '주소는 {address}입니다.')
        msg = self._add_msg_from_dict(result, 'thumUrl', msg, '> 사진보기 : {thumUrl}')
        msg = msg.format(location=location, context=result['context'], category=result['category'],
                         name=result['name'], address=result['address'], thumUrl=result['thumUrl'])

        return msg
