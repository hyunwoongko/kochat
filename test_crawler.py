from _crawler.crawler.dust_crawler import DustCrawler
from _crawler.crawler.retaurant_crawler import RestaurantCrawler
from _crawler.crawler.travel_crawler import TravelCrawler
from _crawler.crawler.weather_crawler import WeatherCrawler

print(DustCrawler().crawl_debug("고성", "모레"), end='\n\n')
print(RestaurantCrawler().crawl_debug("부산", "돼지고기"), end='\n\n')
print(WeatherCrawler().crawl_debug("서울", "이번주"), end='\n\n')
print(TravelCrawler().crawl("부산", "바닷가"), end='\n\n')
