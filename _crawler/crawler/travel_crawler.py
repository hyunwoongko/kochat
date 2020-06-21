"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _crawler.answerer.travel_answerer import TravelAnswerer
from _crawler.crawler.base.crawler import Crawler
from _crawler.editor.travel_editor import TravelEditor
from _crawler.searcher.travel_searcher import TravelSearcher


class TravelCrawler(Crawler):

    def __init__(self):
        self.searcher = TravelSearcher()
        self.editor = TravelEditor()
        self.answerer = TravelAnswerer()

    def crawl(self, location, travel):
        result = self.searcher.search_naver_map(location, travel)
        result = self.editor.edit_travel(location, travel, result)
        result = self.answerer.travel_form(location, travel, result)
        return result
