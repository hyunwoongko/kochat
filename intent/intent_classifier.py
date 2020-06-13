import pandas as pd
import torch
from torch import nn

from config import Config
from util.dataset import Dataset
from util.tokenizer import Tokenizer


class IntentClassifier:
    """
    리트리벌 모델을 사용하지 않고 학습된 Softmax 분류기를 이용해 분류합니다.
    Out of distribution 능력이 없습니다.
    calibrate 되지 않은 softmax score와 metric learning의
    인텐트 검출시 ood 탐지 능력을 비교하기 위해 구현하였습니다.
    """

    def __init__(self, embed, model):
        self.conf = Config()
        self.tok = Tokenizer()
        self.pad_sequencing = Dataset().pad_sequencing
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
