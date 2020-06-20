from _crawler.crawler.base.crawler import Crawler
from _crawler.decorators import answerer


@answerer
class Answerer(Crawler):

    def sorry(self, text=None):
        if text is None:
            return self.fallback
        else:
            return text

    def make_josa(self, a, b, r1, r2):
        if r1 == r2:
            return a
        else:
            return b
