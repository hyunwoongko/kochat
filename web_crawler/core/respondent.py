from web_crawler.base.base_manager import WebCrawlerManager


class Respondent(WebCrawlerManager):
    def dust(self, date, location, results):
        greeting = "{location}의 다양한 대기오염 정보를 전달해드립니다 😊\n\n"
        greeting = [greeting.format(location=location)]

        pattern = "{date} 오전 {kinds} 상태는 {0}입니다. {1}\n" \
                  "{date} 오후 {kinds} 상태는 {2}입니다. {3}\n"
        pattern = [pattern.format(r[0], r[1], r[2], r[3], date=date, kinds=k)
                   for k, r in zip(self.intent['dust'], results)]

        return ''.join(greeting + pattern)

    def sorry(self):
        return "죄송합니다. 그건 잘 모르는 정보에요."
