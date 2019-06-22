# Author : Hyunwoong
# When : 2019-06-22
# Homepage : github.com/gusdnd852
from src.entity.news.entity_recognizer import get_news_entity
from src.entity.restaurant.entity_recognizer import get_restaurant_entity
from src.entity.song.entity_recognizer import get_song_entity
from src.entity.translate.entity_recognizer import get_translate_entity
from src.entity.weather.entity_recognizer import get_weather_entity
from src.entity.wiki.entity_recognizer import get_wiki_entity
from src.intent.classifier import get_intent
from src.scenario.date import date
from src.scenario.issue import issue
from src.scenario.restaurant import restaurant
from src.scenario.song import song
from src.scenario.time import times
from src.scenario.translate import translate
from src.scenario.weather import weather
from src.scenario.wiki import wiki
from src.scenario.wise import wise
from src.util.hanspell.spell_checker import fix
from src.util.tokenizer import tokenize
from src.scenario.dust import dust


def run():
    while True:
        print('User : ', sep='', end='')
        speech = preprcoess(input())
        print('Preprocessed : ' + speech , sep='')
        intent = get_intent(speech)
        print('Intent : ' + intent , sep='')
        entity = get_entity(intent, speech)
        print('Entity : ' + str(entity) , sep='')
        answer = scenario(intent, entity)
        print('A.I : ' + answer, sep='', end='\n\n')


def preprcoess(speech) -> str:
    speech = fix(speech)
    speech = tokenize(speech)
    speech = fix(speech)
    return speech


def get_entity(intent, speech):
    if intent == '날씨' or intent == '먼지':
        return get_weather_entity(speech, False)
    elif intent == '뉴스':
        return get_news_entity(speech, False)
    elif intent == '음악':
        return get_song_entity(speech, False)
    elif intent == '위키' or intent == '인물':
        return get_wiki_entity(speech, False)
    elif intent == '맛집':
        return get_restaurant_entity(speech, False)
    elif intent == '번역':
        return get_translate_entity(speech, False)
    else:
        return None


def scenario(intent, entity) -> str:
    if intent == '먼지':
        return dust(entity)
    elif intent == '날씨':
        return weather(entity)
    elif intent == '인물' or intent == '위키':
        return wiki(entity)
    elif intent == '명언':
        return wise()
    elif intent == '달력':
        return date()
    elif intent == '시간':
        return times()
    elif intent == '번역':
        return translate(entity)
    elif intent == '이슈':
        return issue()
    elif intent == '음악':
        return song(entity)
    elif intent == '맛집':
        return restaurant(entity)
    else:
        return '그 기능은 아직 준비 중이에요.'
