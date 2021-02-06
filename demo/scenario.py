"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""

from kocrawl.dust import DustCrawler
from kocrawl.weather import WeatherCrawler
from kochat.app import Scenario
# from kocrawl.map import MapCrawler
# from kocrawl.restaurant import RestaurantCrawler
# kocrawl의 Map, Restaurant 모듈이 현재 작동하지 않음.

weather = Scenario(
    intent='weather',
    api=WeatherCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

dust = Scenario(
    intent='dust',
    api=DustCrawler().request_debug,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

# restaurant = Scenario(
#     intent='restaurant',
#     api=RestaurantCrawler().request,
#     scenario={
#         'LOCATION': [],
#         'RESTAURANT': ['유명한']
#     }
# )

# travel = Scenario(
#     intent='travel',
#     api=MapCrawler().request_debug,
#     scenario={
#         'LOCATION': [],
#         'PLACE': ['관광지']
#     }
# )
