from embed.embed_processor import EmbedProcessor
from entity import entity_model
from entity.entity_recognizer import EntityRecognizer

emb = EmbedProcessor()
ett = EntityRecognizer(emb, entity_model)
out = ett.recognize("오늘 부산 날씨 어떠니")
print(out)