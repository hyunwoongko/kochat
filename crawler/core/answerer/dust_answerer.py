from crawler.base.answerer import Answerer
from util.oop import singleton


@singleton
class DustAnswerer(Answerer):

    def new_everyday(self, date, location, results):
        print(results)

    def old_today(self, date, location, results):
        print(results)

    def old_tomorrow(self, date, location, results):
        print(results)

