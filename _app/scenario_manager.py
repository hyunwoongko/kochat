"""
@auther Hyunwoong
@since {6/20/2020}
@see : https://github.com/gusdnd852
"""
from _app.scenarios.dust_scenario import DustScenario
from _app.scenarios.fallback_scenario import FallbackScenario
from _app.scenarios.restaurant_scenario import RestaurantScenario
from _app.scenarios.travel_scenario import TravelScenario
from _app.scenarios.weather_scenario import WeatherScenario


class ScenarioManager:
    """
    STATE : 'SUCCESS', 'FALLBACK', 'REQUIRE_XXX'
    """

    def __init__(self):
        self.dust_scen = DustScenario()
        self.weather_scen = WeatherScenario()
        self.restaurant_scen = RestaurantScenario()
        self.travel_scen = TravelScenario()
        self.fallback_scen = FallbackScenario()

    def apply_scenario(self, text, intent, entity):
        text = text.split(' ')

        if intent == '먼지':
            return self.dust_scen(text, entity)
        elif intent == '날씨':
            return self.weather_scen(text, entity)
        elif intent == '맛집':
            return self.restaurant_scen(text, entity)
        elif intent == '여행지':
            return self.travel_scen(text, entity)
        else:
            return self.fallback_scen()
