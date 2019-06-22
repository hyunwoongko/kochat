# Author : Hyunwoong
# When : 2019-06-22
# Homepage : github.com/gusdnd852

import src.crawler.dust as crawler


def dust(named_entity):
    keyword_group = named_entity[0]
    entity_group = named_entity[1]
    date = []
    location = []

    for k in zip(keyword_group, entity_group):
        if 'DATE' in k[1]:
            date.append(k[0])
        elif 'LOCATION' in k[1]:
            location.append(k[0])

    if len(date) == 0:
        date.append('오늘')

    if len(location) == 0:
        while len(location) == 0:
            print('A.I : ' + '어떤 지역을 알려드릴까요?')
            print('User : ', end='', sep='')
            loc = input()
            if loc is not None and loc.replace(' ', '') != '':
                location.append(loc)

    if '오늘' in date:
        return crawler.today_dust(' '.join(location))
    elif date[0] == '내일':
        return crawler.tomorrow_dust(' '.join(location))
    elif '모레' in date or '내일모레' in date:
        return crawler.after_tomorrow_dust(' '.join(location))
    else:
        return '오늘, 내일, 모레의 미세먼지 상태만 알 수 있어요'
