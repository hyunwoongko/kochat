import re

import numpy as np
from crawler.base.editor import Editor
from crawler.core.answerer.dust_answerer import DustAnswerer
from util.oop import singleton


@singleton
class DustEditor(Editor):
    def new_everyday(self, date, location, results):
        """
        DustRequest에서 이미 커버리지 검사 했기 때문에
        오늘, 내일, 모레 이외의 날짜는 나오지 않음.
        """

        results = np.array(results)
        results = results.reshape(3, 6)

        if date in self.date['today']:
            results = results[:, 0:2]
        elif date in self.date['tomorrow']:
            results = results[:, 2:4]
        else:
            results = results[:, 4:6]

        results = np.array([self.edit(kinds, ment) for kinds, ment
                            in zip(results, self.dust_ment.values())])

        return DustAnswerer().new_everyday(date, location, results)

    def old_today(self, date, location, results):
        print(results)

    def old_tomorrow(self, date, location, results):
        print(results)
