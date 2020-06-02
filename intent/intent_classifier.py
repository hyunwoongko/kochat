import torch

from config import Config


class IntentClassifier:

    def __init__(self, embed):
        self.conf = Config()
        self.embed = embed
        self.model = torch.load(self.conf.intent_storefile)

    def inference(self, user_input):
        self.model.eval()

        embedded_input = self.embed.embed(user_input)
        print(embedded_input)