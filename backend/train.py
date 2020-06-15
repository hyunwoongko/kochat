from backend.base.data_builder import DataBuilder
from backend.core.embed.embed_processor import EmbedProcessor
from backend.core.entity import entity_model
from backend.core.entity.entity_recognizer import EntityRecognizer
from backend.core.entity.train_recognizer import TrainRecognizer
from backend.core.intent import intent_model
from backend.core.intent.intent_classifier import IntentClassifier
from backend.core.intent.train_classifier import TrainClassifier
from backend.core.intent.train_retrieval import TrainRetrieval

data_builder = DataBuilder()
embed_data = data_builder.embed_dataset()
embed = EmbedProcessor()
embed.train_model(embed_data)

intent_data = data_builder.intent_dataset(embed)
intent_dict = data_builder.intent_dict
intent_train = TrainClassifier(intent_model, intent_data, intent_dict)
intent_train.train_model()

intent_data = data_builder.intent_dataset(embed)
intent_dict = data_builder.intent_dict
intent_train = TrainRetrieval(intent_model, intent_data, intent_dict)
intent_train.train_model()

entity_data = data_builder.entity_dataset(embed)
entity_dict = data_builder.entity_dict
entity_train = TrainRecognizer(entity_model, entity_data, entity_dict)
entity_train.train_model()

sequence = data_builder.inference_dataset("오늘 전주 날씨", embed)
intent_dict = data_builder.intent_dict
intent_test = IntentClassifier(intent_model, intent_dict)
print(intent_test.inference_model(sequence))

sequence = data_builder.inference_dataset("오늘 전주 날씨", embed)
entity_dict = data_builder.entity_dict
entity_test = EntityRecognizer(entity_model, entity_dict)
print(entity_test.inference_model(sequence))

