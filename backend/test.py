from backend.data.dataloader import DataLoader
from backend.proc.gensim_processor import GensimProcessor
from backend.model import FastText, ResNet
from backend.proc.intent_classifier import IntentClassifier
from backend.proc.intent_retrieval import IntentRetrieval

loader = DataLoader()

emb_model = GensimProcessor(FastText)
emb_model.train(loader.embed_dataset())

intent_model = IntentClassifier(ResNet, loader.intent_dict)
intent_model.train(loader.intent_dataset(emb_model))

intent_model = IntentRetrieval(ResNet, loader.intent_dict)
intent_model.train(loader.intent_dataset(emb_model))
