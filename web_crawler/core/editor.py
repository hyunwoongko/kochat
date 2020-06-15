from web_crawler.base.base_manager import WebCrawlerManager
from web_crawler.core.respondent import Respondent


class Editor(WebCrawlerManager):
    def __init__(self):
        super().__init__()
        self.speaker = Respondent()

    def dust(self, date, location, results, old=False):
        print(results)
