from base.data_managers.data_builder import DataBuilder
from core.embed.embed_processor import EmbedProcessor
from core.entity import entity_model
from core.entity.train_recognizer import TrainRecognizer
from core.intent import intent_model
from core.intent.intent_classifier import IntentClassifier
from core.intent.train_classifier import TrainClassifier
from core.intent.train_retrieval import TrainRetrieval

data_builder = DataBuilder()
embed_data = data_builder.embed_dataset()
embed = EmbedProcessor()
# embed.train_model(embed_data)
#
# intent_data = data_builder.intent_dataset(embed)
# intent_dict = data_builder.intent_dict
# intent_train = TrainClassifier(intent_model, intent_data, intent_dict)
# intent_train.train_model()
#
intent_data = data_builder.intent_dataset(embed)
intent_dict = data_builder.intent_dict
intent_train = TrainRetrieval(intent_model, intent_data, intent_dict)
intent_train.train_model()
#
# entity_data = data_builder.entity_dataset(embed)
# entity_dict = data_builder.entity_dict
# entity_train = TrainRecognizer(entity_model, entity_data, entity_dict)
# entity_train.train_model()

# while True:
#     intent_data = data_builder.inference_sequence(input(), embed)
#     intent_dict = data_builder.intent_dict
#     intent_test = IntentClassifier(intent_model, intent_dict)
#     output = intent_test.inference_model(intent_data)
#     print(output)

