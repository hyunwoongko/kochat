"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

from configs import GlobalConfigs
from embed.embed_processor import EmbedProcessor
from intent.intent_processor import IntentProcessor
from intent.model import text_cnn

if __name__ == '__main__':
    global_conf = GlobalConfigs()

    emb = EmbedProcessor(
        store_path=global_conf.embed_storepath,
        data_path=global_conf.intent_datapath)

    intent = IntentProcessor(
        emb=emb,
        model=text_cnn,
        store_path="",
        data_path=global_conf.intent_datapath)

    intent.train()
