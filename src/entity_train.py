# Author : Hyunwoong
# When : 2019-06-23
# Homepage : github.com/gusdnd852

from src.entity.news.entity_recognizer import get_news_entity
from src.entity.restaurant.entity_recognizer import get_restaurant_entity
from src.entity.song.entity_recognizer import get_song_entity
from src.entity.translate.entity_recognizer import get_translate_entity
from src.entity.weather.entity_recognizer import get_weather_entity
from src.entity.wiki.entity_recognizer import get_wiki_entity

# Choose entity recognizer you want !!
# 학습을 원하는 개체명인식기를 고르세요 !!
if __name__ == '__main__':
    # get_news_entity('_', is_train=True)
    # get_restaurant_entity('_', is_train=True)
    # get_song_entity('_', is_train=True)
    # get_translate_entity('_', is_train=True)
    # get_weather_entity('_', is_train=True)
    # get_wiki_entity('_', is_train=True)
    pass
