from backend.data.dataset import Dataset
from backend.loss.center_loss import CenterLoss
from backend.loss.cosface import CosFace
from backend.model.embed_fasttext import EmbedFastText
from backend.model.intent_bilstm import IntentBiLSTM
from backend.model.intent_resnet import IntentResNet
from backend.proc.gensim_processor import GensimProcessor
from backend.proc.intent_classifier import IntentClassifier
from backend.proc.intent_retrieval import IntentRetrieval

dataset = Dataset()

emb_model = EmbedFastText()
emb_proc = GensimProcessor(emb_model)
emb_proc.train(dataset.embed_dataset())

# intent_model = IntentResNet(dataset.intent_dict)
intent_model = IntentBiLSTM(dataset.intent_dict)
intent_proc = IntentClassifier(intent_model)
intent_proc.train(dataset.intent_dataset(emb_proc))

inference_data = dataset.inference_dataset("오늘 전주 날씨 알려줘", emb_proc)
print(intent_proc.inference(inference_data))

intent_model = IntentResNet(dataset.intent_dict)
loss = CenterLoss(dataset.intent_dict)
intent_proc = IntentRetrieval(loss=loss, model=intent_model)
intent_proc.train(dataset.intent_dataset(emb_proc))
