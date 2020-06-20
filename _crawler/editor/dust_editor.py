import numpy as np
from _crawler.editor.base.editor import Editor
from _crawler.answerer.dust_answerer import DustAnswerer


class DustEditor(Editor):
    """
    DustRequest에서 이미 커버리지 검사 했기 때문에
    오늘, 내일, 모레 이외의 날짜는 나오지 않음.
    """

    def new_everyday(self, date, location, results):
        results = np.array(results)
        results = results.reshape(3, 6)

        if date in self.date['today']:
            results = results[:, 0:2]  # 오늘 오전, 오후
        elif date in self.date['tomorrow']:
            results = results[:, 2:4]  # 내일 오전, 오후
        else:
            results = results[:, 4:6]  # 모레 오전, 오후

        josa = self.make_josa('는', '도', self.flatten(results))
        morning = [ment[kinds] for kinds, ment in zip(results[:, 0], self.dust.values())]
        afternoon = [ment[kinds] for kinds, ment in zip(results[:, 1], self.dust.values())]
        ments = np.concatenate([np.expand_dims(morning, 0), np.expand_dims(afternoon, 0)]).T
        return np.concatenate(ments), josa

    def old_today(self, date, location, results):
        fine_dust = results[6]  # 미세먼지
        ultra_fine = results[0]  # 초미세먼지
        ozon = results[1]  # 오존

        results = [fine_dust, ultra_fine, ozon]
        josa = self.make_josa('는', '도', results)
        ments = [ment[kinds] for kinds, ment in zip(results, self.dust.values())]
        return ments, josa

    def old_tomorrow(self, date, location, results):
        if date in self.date['tomorrow']:
            # 내일 오전, 오후
            fine = results[0:2]
            ultra_fine = results[4:6]
            ozon = results[8:10]
        else:
            # 모레 오전, 오후
            fine = results[2:4]
            ultra_fine = results[6:8]
            ozon = results[10:12]

        results = [fine, ultra_fine, ozon]
        josa = self.make_josa('는', '도', np.concatenate(results))

        results = np.array(results).T
        morning, afternoon = results[0], results[1]
        morning = [ment[kinds] for kinds, ment in zip(morning, self.dust.values())]
        afternoon = [ment[kinds] for kinds, ment in zip(afternoon, self.dust.values())]
        ments = np.concatenate([np.expand_dims(morning, 0), np.expand_dims(afternoon, 0)]).T
        return np.concatenate(ments), josa
