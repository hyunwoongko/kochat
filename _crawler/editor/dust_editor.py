import re

from _crawler.editor.base.editor import Editor


class DustEditor(Editor):

    def edit_morning_afternoon(self, location: str, date: str, results: dict) -> tuple:
        """
        오전-오후 형식의 데이터 딕셔너리를 수정합니다.

        :param location: 지역
        :param date: 날짜
        :param results: 입력 딕셔너리
        :return: 수정된 딕셔너리
        """

        data_dict = {'morning_fine_dust': None,
                     'afternoon_fine_dust': None,
                     'morning_ultra_dust': None,
                     'afternoon_ultra_dust': None,
                     'morning_ozon': None,
                     'afternoon_ozon': None}

        result_list = []

        # 날짜에 따라 딱 6개씩만 결과를 넣음
        for k, v in results.items():
            if date in self.date['today'] and 'today_' in k:
                result_list.append(v)
            elif date in self.date['tomorrow'] and 'tomorrow_' in k:
                result_list.append(v)
            elif date in self.date['after'] and 'after_' in k:
                result_list.append(v)

        for k, r in zip(data_dict.keys(), result_list):
            dust_state = re.sub(' ', '', r)
            data_dict[k] = self.dust[dust_state]

        josa = self.enumerate_josa('는', '도', result_list)
        return data_dict, josa

    def edit_single(self, location: str, date: str, results: dict) -> tuple:
        """
        오전-오후 구분이 없는 형식의 딕셔너리를 수정합니다.

        :param location: 지역
        :param date: 날짜
        :param results: 입력 딕셔너리
        :return: 수정된 딕셔너리
        """

        for k, v in results.items():
            if v is not None:
                dust_state = re.sub(' ', '', v)
                results[k] = self.dust[dust_state]

        josa = self.enumerate_josa('는', '도', list(results.values()))
        return results, josa
