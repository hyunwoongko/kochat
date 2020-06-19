import os
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from backend.data.dataset import Dataset
from backend.data.preprocessor import Preprocessor
from backend.loss.center_loss import CenterLoss
from backend.loss.crf_loss import CRFLoss
from backend.model.embed_fasttext import EmbedFastText
from backend.model.entity_lstm import EntityLSTM
from backend.model.intent_cnn import IntentCNN
from backend.model.intent_lstm import IntentLSTM
from backend.proc.entity_recognizer import EntityRecognizer
from backend.proc.gensim_embedder import GensimEmbedder
from backend.proc.intent_classifier import IntentClassifier
from backend.proc.intent_retrival import IntentRetrieval

# 1. 데이터 셋 객체를 생성합니다. (ood 여부 선택)
dataset = Dataset(Preprocessor(), ood=True)

# 2. 임베딩 프로세서를 학습합니다.
embed_dataset = dataset.load_embed()
embed_processor = GensimEmbedder(
    model=EmbedFastText())
embed_processor.fit(embed_dataset)

# 3. 인텐트 분류기를 학습합니다. (분류기와 검색기 중 택 1)
intent_dataset = dataset.load_intent(embed_processor)
intent_processor = IntentClassifier(
    model=IntentLSTM(dataset.intent_dict))
intent_processor.fit(intent_dataset)

# 4. 인텐트 검색기를 학습합니다 (분류기와 검색기 중 택 1)
intent_dataset = dataset.load_intent(embed_processor)
intent_processor = IntentRetrieval(
    model=IntentCNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict))
intent_processor.fit(intent_dataset)

# 5. 엔티티 검출기를 학습합니다.
entity_dataset = dataset.load_entity(embed_processor)
entity_processor = EntityRecognizer(
    model=EntityLSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict))
entity_processor.fit(entity_dataset)
