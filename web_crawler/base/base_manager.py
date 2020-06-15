from web_crawler import config


class WebCrawlerManager:
    def __init__(self):
        for key, val in config.DATE.items():
            setattr(self, key, val)
        for key, val in config.INTENT.items():
            setattr(self, key, val)


class SearchManager:
    def __init__(self):
        for key, val in config.URLS.items():
            setattr(self, key, val)
