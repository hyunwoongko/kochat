from _crawler.crawler.base.crawler import Crawler
from _crawler.decorators import answerer


@answerer
class Answerer(Crawler):

    def add_msg_from_dict(self, dict_, key, msg, insert):
        if len(dict_[key]) > 0:
            if dict_[key][0] is not None:
                msg += insert + ' '
        return msg

    def sorry(self, text=None):
        if text is None:
            return self.fallback
        else:
            return text
