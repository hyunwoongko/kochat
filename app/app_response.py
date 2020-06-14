from base.base_component import Crawler


class UserResponse(Crawler):
    def dust_(self, date, location, params):
        if date in self.date_['today']:
            params = params[0:4]
        elif date in self.date_['tomorrow']:
            params = params[4:8]
        elif date in self.date_['after']:
            params = params[0:4]

        msg = "{0}의 {1} 미세먼지를 알려드릴게요.\n" \
              "오전의 미세먼지 농도는 {2}이며 {3}\n" \
              "오후의 미세먼지 농도는 {4}이며 {5}" \
            .format(location, date, params[0], params[1], params[2], params[3])
        return msg + '꼭 마스크를 착용하세요' if '나쁨' in params else msg

    def sorry_(self):
        return "죄송합니다. 제가 아직 잘 모르는 말이에요.\n" \
               "죄송하지만 조금만 더 명확하게 말씀해주시겠어요?"
