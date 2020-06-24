import os
import warnings

from _backend.data.utils.dataset import Dataset
from _backend.loss.center_loss import CenterLoss
from _backend.loss.crf_loss import CRFLoss
from _backend.loss.cross_entropy_loss import CrossEntropyLoss
from _backend.model.embed_fasttext import EmbedFastText
from _backend.model.entity_lstm import EntityLSTM
from _backend.model.intent_cnn import IntentCNN
from _backend.proc.base.gensim_processor import GensimProcessor
from _backend.proc.entity_recognizer import EntityRecognizer
from _backend.proc.intent_classifier import IntentClassifier
from _backend.proc.softmax_classifier import SoftmaxClassifier

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 1. 데이터 셋 객체를 생성합니다. (ood 여부 선택)
dataset = Dataset(ood=True)

# 2. 임베딩 프로세서를 학습합니다.
embed = GensimProcessor(
    model=EmbedFastText()
)
# embed.fit(dataset.load_embed())

# 3. 의도 분류기를 학습합니다
intent = SoftmaxClassifier(
    model=IntentCNN(dataset.intent_dict)
)
intent.fit(dataset.load_intent(embed))

entity = EntityRecognizer(
    model=EntityLSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)
# entity.fit(dataset.load_entity(embed))

intent = IntentClassifier(
    model=IntentCNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict)
)

# intent.fit(dataset.load_intent(embed))