import pandas as pd
import torch
from torch import nn

from config import Config
from util.dataset import Dataset
from util.tokenizer import Tokenizer


class EntityRecognizer:

    def __init__(self, embed, model):
        self.conf = Config()
        self.data = Dataset()
        self.tok = Tokenizer()
        self.model = model.Net(len(self.data.label_dict)).cuda()
        self.model.load_state_dict(torch.load(self.conf.entity_storefile))
        self.model.eval()
        self.embed = embed
        self.softmax = nn.Softmax()
        self.label_set = self.data.label_dict()

    def recognize(self, text):
        tokenized = self.tok.tokenize(text)
        embedded = self.embed.embed(tokenized)
        sequence = self.data.pad_sequencing(embedded)
        sequence = sequence.unsqueeze(0).cuda()

        output = self.model(sequence.permute(0, 2, 1)).float()
        output = self.model.classifier(output.squeeze())
        output = self.softmax(output)
        _, predict = torch.max(output, dim=0)

        report = []
        for i, o in enumerate(output):
            label = self.label_set[i]
            logit = round(o.item(), self.conf.logging_precision)
            report.append((label, logit))

        print(report)
        return self.label_set[predict.item()]
