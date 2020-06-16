from crawler.base.searcher import Searcher
from util.oop import singleton


@singleton
class DustSearcher(Searcher):
    CSS = {
        'new_everyday': '.on.now > em',
        'old_today': 'div.all_state > ul.state_list > li > span.state_info',
        'old_tomorrow': 'div.air_detail > div.tb_scroll > table > tbody > tr > td'
    }

    def _make_query(self, date, location):
        return [' '.join([date, location] + [i])
                for i in self.kinds['dust']]

    def new_everyday(self, date, location):
        query = self._make_query(date, location)
        result = [self._naver(q, self.CSS['new_everyday']) for q in query]
        result = self.flatten(self.flatten(result))
        return result

    def old_today(self, date, location):
        query = self._make_query(date, location)
        result = [self._naver(q, self.CSS['old_today']) for q in query]
        result = [i[0].strip() for i in self.flatten(result)]
        return result

    def old_tomorrow(self, date, location):
        query = self._make_query(date, location)
        result = [self._naver(q, self.CSS['old_tomorrow']) for q in query]
        result = self.flatten(self.flatten(result))
        result = [self.untag(i) for i in result]
        return result
