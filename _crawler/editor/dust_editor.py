import numpy as np
from _crawler.editor.base.editor import Editor
from _crawler.answerer.dust_answerer import DustAnswerer


class DustEditor(Editor):

    def edit_new_everyday(self, location, date, results):
        results = np.array(results)
        results = results.reshape(3, 6)

        if date in self.date['today']:
            results = results[:, 0:2]  # 오늘 오전, 오후
        elif date in self.date['tomorrow']:
            results = results[:, 2:4]  # 내일 오전, 오후
        else:  # after_tomorrow
            results = results[:, 4:6]  # 모레 오전, 오후

        josa = self.enumerate_josa('는', '도', self._flatten_list(results))
        morning = [ment[kinds] for kinds, ment in zip(results[:, 0], self.dust.values())]
        afternoon = [ment[kinds] for kinds, ment in zip(results[:, 1], self.dust.values())]
        ments = np.concatenate([np.expand_dims(morning, 0), np.expand_dims(afternoon, 0)]).T
        return np.concatenate(ments), josa

    def edit_old_today(self, location, date, results):
        fine_dust = results[6]  # 미세먼지
        ultra_fine = results[0]  # 초미세먼지
        ozon = results[1]  # 오존

        results = [fine_dust, ultra_fine, ozon]
        josa = self.enumerate_josa('는', '도', results)
        ments = [ment[kinds] for kinds, ment in zip(results, self.dust.values())]
        return ments, josa

    def edit_old_tomorrow(self, location, date, results):


        if date in self.date['tomorrow']:
            # 내일 오전, 오후
            fine = results[0:2]
            ultra_fine = results[4:6]
            ozon = results[8:10]
        else:  # after_tomorrow
            # 모레 오전, 오후
            fine = results[2:4]
            ultra_fine = results[6:8]
            ozon = results[10:12]

        results = [fine, ultra_fine, ozon]
        josa = self.enumerate_josa('는', '도', np.concatenate(results))

        results = np.array(results).T
        morning, afternoon = results[0], results[1]

        morning = [ment[kinds] for kinds, ment in zip(morning, self.dust.values())]
        afternoon = [ment[kinds] for kinds, ment in zip(afternoon, self.dust.values())]
        ments = np.concatenate([np.expand_dims(morning, 0), np.expand_dims(afternoon, 0)]).T
        return np.concatenate(ments), josa
