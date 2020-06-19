from backend.data.dataset import Dataset
from backend.data.preprocessor import Preprocessor
from backend.loss.center_loss import CenterLoss
from backend.loss.crf_loss import CRFLoss
from backend.loss.softmax_loss import SoftmaxLoss
from backend.model.embed_fasttext import EmbedFastText
from backend.model.entity_lstm import EntityLSTM
from backend.model.intent_cnn import IntentCNN
from backend.proc.entity_recognizer import EntityRecognizer
from backend.proc.gensim_embedder import GensimEmbedder
from backend.proc.intent_retrival import IntentRetrieval

dataset = Dataset(Preprocessor(), ood=True)
embed_dataset = dataset.load_embed()
embed_processor = GensimEmbedder(EmbedFastText())
# embed_processor.fit(embed_dataset)

intent_dataset = dataset.load_intent(embed_processor)
intent_processor = IntentRetrieval(
    model=IntentCNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict))

intent_processor.fit(intent_dataset)

# while True:
#     print("입력하세요 : ", end='')
#     input_sequence = dataset.load_predict(input(), embed_processor)
#     print(intent_processor.predict(input_sequence, calibrate=True))

# entity_dataset = dataset.load_entity(embed_processor)
# entity_processor = EntityRecognizer(
#     model=EntityLSTM(dataset.entity_dict),
#     loss=SoftmaxLoss(dataset.entity_dict))
#
# entity_processor.fit(entity_dataset)
