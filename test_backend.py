import os
import warnings

from _backend.data.utils.dataset import Dataset
from _backend.data.utils.organizer import Organizer
from _backend.loss.center_loss import CenterLoss
from _backend.loss.crf_loss import CRFLoss
from _backend.model.embed_fasttext import EmbedFastText
from _backend.model.entity_lstm import EntityLSTM
from _backend.model.intent_cnn import IntentCNN
from _backend.proc.entity_recognizer import EntityRecognizer
from _backend.proc.gensim_embedder import GensimEmbedder
from _backend.proc.intent_classifier import IntentClassifier

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 1. 데이터 셋 객체를 생성합니다. (ood 여부 선택)
dataset = Dataset(ood=True)


# 2. 임베딩 프로세서를 학습합니다.
embed_dataset = dataset.load_embed()
embed_processor = GensimEmbedder(
    model=EmbedFastText())
# embed_processor.fit(embed_dataset)

# 3. 의도 분류기를 학습합니다
intent_dataset = dataset.load_intent(embed_processor)
intent_processor = IntentClassifier(
    model=IntentCNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict))
intent_processor.fit(intent_dataset)

# 4. 개체명 인식기를 학습합니다.
entity_dataset = dataset.load_entity(embed_processor)
entity_processor = EntityRecognizer(
    model=EntityLSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict))
entity_processor.fit(entity_dataset)
