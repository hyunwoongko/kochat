# Author : Hyunwoong
# When : 2019-06-22
# Homepage : github.com/gusdnd852

import src.crawler.youtube as crawler


def song(entity):
    keyword_group = entity[0]
    entity_group = entity[1]

    words_list = []

    for k in zip(keyword_group, entity_group):
        if k[1] != 'O':
            words_list.append(k[0])
    if len(words_list) == 0:
        words_list.append('듣기 좋은 노래')
    return crawler.get_youtube(' '.join(words_list))
