from _crawler.searcher.base.searcher import Searcher


class DustSearcher(Searcher):
    CSS = {
        'new_everyday': '.on.now > em',
        'old_today': 'div.all_state > ul.state_list > li > span.state_info',
        'old_tomorrow': 'div.air_detail > div.tb_scroll > table > tbody > tr > td'
    }

    def _make_query(self, location, date):
        return [' '.join([location, date] + [i])
                for i in self.kinds['dust']]

    def new_everyday(self, location, date):
        query = self._make_query(location, date)
        result = [self._bs4_contents(self.url['naver'],
                                     selectors=[self.CSS['new_everyday']],
                                     query=q) for q in query]
        result = self._flatten_list(result)
        return result

    def old_today(self, location, date):
        query = self._make_query(location, date)
        result = [self._bs4_contents(self.url['naver'],
                                     selectors=[self.CSS['old_today']],
                                     query=q) for q in query]

        fine = [i[0].strip() for i in result[0]]
        ultra_fine = [i[0].strip() for i in result[1]]
        ozon = [i[0].strip() for i in result[2]]

        result = [fine, ultra_fine, ozon]
        result = self._flatten_list(result)
        return result

    def old_tomorrow(self, location, date):
        query = self._make_query(location, date)
        result = [self._bs4_contents(self.url['naver'],
                                     selectors=[self.CSS['old_tomorrow']],
                                     query=q) for q in query]

        fine = [self._untag(i[0]) for i in result[0]]
        ultra_fine = [self._untag(i[0]) for i in result[1]]
        ozon = [self._untag(i[0]) for i in result[2]]

        result = [fine, ultra_fine, ozon]
        result = self._flatten_list(result)
        return result
