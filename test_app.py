"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from _app.restful_api import KochatApi
from _backend.data.dataset import Dataset
from _backend.data.preprocessor import Preprocessor
from _backend.loss.center_loss import CenterLoss
from _backend.loss.crf_loss import CRFLoss
from _backend.model.embed_fasttext import EmbedFastText
from _backend.model.entity_lstm import EntityLSTM
from _backend.model.intent_cnn import IntentCNN
from _backend.proc.entity_recognizer import EntityRecognizer
from _backend.proc.gensim_embedder import GensimEmbedder
from _backend.proc.intent_classifier import IntentClassifier

if __name__ == '__main__':
    dataset = Dataset(
        preprocessor=Preprocessor(),
        ood=True
    )

    embed_processor = GensimEmbedder(
        model=EmbedFastText()
    )

    intent_classifier = IntentClassifier(
        model=IntentCNN(dataset.intent_dict),
        loss=CenterLoss(dataset.intent_dict)
    )

    entity_recognizer = EntityRecognizer(
        model=EntityLSTM(dataset.entity_dict),
        loss=CRFLoss(dataset.entity_dict)
    )

    kochat_api = KochatApi(
        dataset=dataset,
        embed_processor=embed_processor,
        intent_classifier=intent_classifier,
        entity_recognizer=entity_recognizer
    )

    kochat_api.run(
        ip='0.0.0.0',
        port=9893
    )
