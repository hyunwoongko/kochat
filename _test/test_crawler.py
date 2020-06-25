from _crawler.crawler.dust_crawler import DustCrawler
from _crawler.crawler.retaurant_crawler import RestaurantCrawler
from _crawler.crawler.travel_crawler import TravelCrawler
from _crawler.crawler.weather_crawler import WeatherCrawler

# 미세먼지 신버전 테스트
print(DustCrawler().crawl_debug("서울", "오늘"), end='\n\n')
print(DustCrawler().crawl_debug("서울", "내일"), end='\n\n')
print(DustCrawler().crawl_debug("서울", "모레"), end='\n\n')

# 미세먼지 구버전 테스트
print(DustCrawler().crawl_debug("고성", "오늘"), end='\n\n')
print(DustCrawler().crawl_debug("고성", "내일"), end='\n\n')
print(DustCrawler().crawl_debug("고성", "모레"), end='\n\n')

# 맛집 테스트
print(RestaurantCrawler().crawl_debug("대구", "고깃집"), end='\n\n')
print(RestaurantCrawler().crawl_debug("전주", "비빔밥"), end='\n\n')

# 네이버 날씨 테스트
print(WeatherCrawler().crawl_debug("서울", "오늘"), end='\n\n')
print(WeatherCrawler().crawl_debug("서울", "내일"), end='\n\n')
print(WeatherCrawler().crawl_debug("서울", "모레"), end='\n\n')

# 구글 날씨 테스트
print(WeatherCrawler().crawl_debug("서울", "수요일"), end='\n\n')
print(WeatherCrawler().crawl_debug("서울", "6월 28일"), end='\n\n')
print(WeatherCrawler().crawl_debug("서울", "이번주"), end='\n\n')

# 여행지 테스트
print(TravelCrawler().crawl_debug("부산", "바닷가"), end='\n\n')
print(TravelCrawler().crawl_debug("서울", "도심"), end='\n\n')
print(TravelCrawler().crawl_debug("전주", "한옥마을"), end='\n\n')
