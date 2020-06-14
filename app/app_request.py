import urllib
from urllib.request import urlopen, Request

import bs4

from app.app_response import UserResponse


class UserRequest(UserResponse):
    """
    이전 버전에서 너무 난잡하게 구현했었음.
    Selector 잘 구성해서 최대한 간결하고 깔끔하게!!
    """

    def dust(self, location, date):
        result = [s.string for s in self.__connect_naver(
            location + " " + date + "미세먼지"
        ).select('.detail_info > dl > dd')]

        return self.dust_(date, location, result)

    def weather(self, location, date):
        pass

    def __connect_naver(self, query):
        url = self.naver + urllib.parse.quote(query)
        return bs4.BeautifulSoup(urlopen(Request(url)).read(), 'html.parser')
