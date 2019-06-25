# Author : Hyunwoong
# When : 2019-06-23
# Homepage : github.com/gusdnd852

from src.crawler import restaurant as crawler

def restaurant(named_entity):
    keyword_group = named_entity[0]
    entity_group = named_entity[1]
    location = []

    for k in zip(keyword_group, entity_group):
        if 'LOCATION' in k[1]:
            location.append(k[0])

    if len(location) == 0:
        while len(location) == 0:
            print('A.I : ' + '어떤 맛집을 알려드릴까요?')
            print('User : ', end='', sep='')
            loc = input()
            if loc is not None and loc.replace(' ', '') != '':
                location.append(loc)

    return crawler.recommend_restaurant(' '.join(location))
