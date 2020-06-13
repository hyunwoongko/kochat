from embed.embed_processor import EmbedProcessor
from entity import entity_model
from entity.entity_recognizer import EntityRecognizer
from util.dataset import Dataset

embed = EmbedProcessor()
entity_recognizer = EntityRecognizer(embed, entity_model)
output = entity_recognizer.recognize("전주 날씨 알려줘")
print(output)
