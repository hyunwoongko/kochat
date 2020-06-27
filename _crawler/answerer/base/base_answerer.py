from _crawler.crawler.base.base_crawler import BaseCrawler
from _crawler.decorators import answerer


@answerer
class BaseAnswerer(BaseCrawler):

    def _add_msg_from_dict(self, dict_: dict, key: str, msg: str, insert: str) -> str:
        """
        딕셔너리에 있는 말을 메시지 문자열에 통합합니다.

        :param dict_: 데이터 딕셔너리
        :param key: 키값
        :param msg: 지금까지 만든 문자열 객체
        :param insert: 뒤에 덧붙일 말
        :return: 딕셔너리의 값이 추가된 문자열 객체
        """

        if len(dict_[key]) > 0:
            if dict_[key][0] is not None:
                msg += insert + ' '
        return msg

    def sorry(self, text: str = None) -> str:
        """
        fallback 메시지를 출력합니다.
        
        :param text: fallback 메시지 (없으면 default 출력)
        :return: fallback 메시지
        """

        if text is None:
            return self.fallback
        else:
            return text
