# Author : Hyunwoong
# When : 2019-06-22
# Homepage : github.com/gusdnd852

from src.crawler.wiki import wiki as crawler


def wiki(entity):
    keyword_group = entity[0]
    entity_group = entity[1]

    words_list = []

    for k in zip(keyword_group, entity_group):
        if k[1] != 'O':
            words_list.append(k[0])
    if len(words_list) == 0:
        print('A.I : ' + '어떤 말을 알려드릴까요?', end='')
        print('User : ')
        words_list.append(input())
    return crawler(' '.join(words_list))
