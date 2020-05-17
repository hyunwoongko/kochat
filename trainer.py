"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import torch

from configs import GlobalConfigs, ResNetConfigs
from embedding.embedding import Embedding
from embedding.visualization import EmbeddingVisualizer
from intent.intent_trainer import IntentTrainer
from intent.model.text_cnn import Model
from torchsummary import summary


class Trainer:
    global_conf = GlobalConfigs()

    def train_embed(self):
        emb = Embedding(store_path=self.global_conf.root_path + "models\\fasttext")
        emb.train(data_path=self.global_conf.intent_path)
        emb_vis = EmbeddingVisualizer()
        emb_vis.visualize(emb=emb)

    def train_intent(self, model, model_config):
        IntentTrainer(model, model_config=model_config, data_path=self.global_conf.intent_path)()

    def train_entity(self):
        pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_intent(model=Model(),
                         model_config=ResNetConfigs())
