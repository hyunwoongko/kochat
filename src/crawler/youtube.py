import random
import urllib
from urllib.request import urlopen

import bs4


def get_youtube(song_name):
    parsed_song = urllib.parse.quote(song_name + ' 노래 mp3 듣기')
    url = 'https://www.youtube.com/results?sp=EgIYAQ%253D%253D&search_query=' + parsed_song
    html_doc = urlopen(url)
    soup = bs4.BeautifulSoup(html_doc, 'html.parser')
    link = soup.findAll('div', attrs={'class': 'yt-lockup-dismissable'})
    idx = random.randint(0, 8)
    link = 'https://www.youtube.com/' + link[idx].find('h3').find('a')['href']

    # _filename = 'song'
    # YouTube(link).streams.first().download(filename=_filename)
    # mp4 다운로드 코드 
    return link
