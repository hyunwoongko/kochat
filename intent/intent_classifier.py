import pandas as pd
import torch
from torch import nn

from config import Config
from util.dataset import Dataset
from util.tokenizer import Tokenizer


class IntentClassifier:
    conf = Config()
    tok = Tokenizer()
    pad_sequencing = Dataset().pad_sequencing

    def __init__(self, embed, model):
        self.model = model.Net().cuda()
        self.model.load_state_dict(torch.load(self.conf.intent_storefile))
        self.model.eval()
        self.embed = embed
        self.softmax = nn.Softmax()
        self.label_set = self.make_intent_map()

    def make_intent_map(self):
        data = pd.read_csv(self.conf.intent_datapath).values.tolist()
        label_set = []
        for d in data:
            label = d[1]
            if label not in label_set:
                label_set.append(label)

        return label_set

    def classify(self, text):
        tokenized = self.tok.tokenize(text)
        embedded = self.embed.embed(tokenized)
        sequence = self.pad_sequencing(embedded)
        sequence = sequence.unsqueeze(0).cuda()

        output = self.model(sequence.permute(0, 2, 1)).float()
        output = self.model.out(output.squeeze())
        output = self.softmax(output)
        _, predict = torch.max(output, dim=0)

        report = []
        for i, o in enumerate(output):
            label = self.label_set[i]
            logit = round(o.item(), self.conf.intent_log_precision)
            report.append((label, logit))

        print(report)
        return self.label_set[predict.item()]
