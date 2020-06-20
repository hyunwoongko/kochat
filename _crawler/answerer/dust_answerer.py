from _crawler.answerer.base.answerer import Answerer


class DustAnswerer(Answerer):

    def new_everyday(self, date, location, results, josa):
        msg = self.dust.format(location=location)
        msg += '{date} 오전 미세먼지 상태{j0} {r0} 오후 상태{j1} {r1} ' \
               '오전 초미세먼지 상태{j2} {r2} 오후 상태{j3} {r3} ' \
               '오전 대기중 오존 상태{j4} {r4} 오후 상태{j5} {r5} ' \
            .format(date=date,
                    j0=josa[0], r0=results[0], j1=josa[1], r1=results[1],
                    j2=josa[2], r2=results[2], j3=josa[3], r3=results[3],
                    j4=josa[4], r4=results[4], j5=josa[5], r5=results[5])

        return msg

    def old_today(self, date, location, results, josa):
        msg = self.dust.format(location=location)
        msg += '{date} 미세먼지 상태{j0} {r0} ' \
               '초미세먼지 상태{j1} {r1} ' \
               '대기중 오존 상태{j2} {r2} ' \
            .format(date=date,
                    j0=josa[0], r0=results[0],
                    j1=josa[1], r1=results[1],
                    j2=josa[2], r2=results[2])

        return msg

    def old_tomorrow(self, date, location, results, josa):
        msg = self.dust.format(location=location)
        msg += '{date} 오전 미세먼지 상태{j0} {r0} 오후 상태{j1} {r1} ' \
               '오전 초미세먼지 상태{j2} {r2} 오후 상태{j3} {r3} ' \
               '오전 대기중 오존 상태{j4} {r4} 오후 상태{j5} {r5} ' \
            .format(date=date,
                    j0=josa[0], r0=results[0], j1=josa[1], r1=results[1],
                    j2=josa[2], r2=results[2], j3=josa[3], r3=results[3],
                    j4=josa[4], r4=results[4], j5=josa[5], r5=results[5])

        return msg
