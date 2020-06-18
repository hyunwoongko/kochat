from backend.data.dataset import Dataset
from backend.data.preprocessor import Preprocessor
from backend.loss.center_loss import CenterLoss
from backend.model.embed_fasttext import EmbedFastText
from backend.model.intent_resnet import IntentResNet
from backend.proc.fallback_detector import FallbackDetector
from backend.proc.gensim_embedder import GensimEmbedder
from backend.proc.intent_retrival import IntentRetrieval

dataset = Dataset(Preprocessor(), ood=True)

embed_dataset = dataset.embed_dataset()
gensim_emb = GensimEmbedder(EmbedFastText())
# gensim_emb.train(embed_dataset)

intent_retrieval = IntentRetrieval(
    model=IntentResNet(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict),
    fallback=FallbackDetector())

intent_retrieval.train(dataset.intent_dataset(gensim_emb))
intent_retrieval.test()

while True:
    data = dataset.inference_dataset(input(), gensim_emb)
    output = intent_retrieval.inference(data, calibrate=True)
    print(output)