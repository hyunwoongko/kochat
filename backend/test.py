from backend.data.dataloader import DataLoader
from backend.loss.center_loss import CenterLoss
from backend.model.embed_fasttext import EmbedFastText
from backend.model.intent_bilstm import IntentBiLSTM
from backend.model.intent_resnet import IntentResNet
from backend.proc.gensim_processor import GensimProcessor
from backend.proc.intent_classifier import IntentClassifier
from backend.proc.intent_retrieval import IntentRetrieval

loader = DataLoader()

# embedding training
emb_model = EmbedFastText()
emb_proc = GensimProcessor(emb_model)
emb_proc.train(loader.embed_dataset())

# intent classification training
intent_model = IntentResNet(loader.intent_dict)
# intent_model = IntentBiLSTM(loader.intent_dict)
intent_proc = IntentClassifier(intent_model)
intent_proc.train(loader.intent_dataset(emb_proc))

inference_data = loader.inference_dataset("오늘 전주 날씨 알려줘", emb_proc)
print(intent_proc.inference(inference_data))

loss = CenterLoss(loader.intent_dict)
intent_proc = IntentRetrieval(
    loss=loss,
    model=intent_model)
intent_proc.train(loader.intent_dataset(emb_proc))
