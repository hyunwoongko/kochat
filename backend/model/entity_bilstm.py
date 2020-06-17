"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn

from backend.decorators import entity, model


@entity
@model
class EntityBiLSTM(nn.Module):

    def __init__(self, label_dict):
        super().__init__()
        self.classes = len(label_dict)
        self.direction = 2  # bidirectional
        self.lstm = nn.LSTM(input_size=self.vector_size,
                            hidden_size=self.d_model,
                            num_layers=self.layers,
                            batch_first=True,
                            bidirectional=True if self.direction == 2 else False)

        self.classifier = nn.Linear(self.d_model * self.direction, self.classes)

    def init_hidden(self, batch_size):
        param1 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        param2 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        return torch.autograd.Variable(param1), torch.autograd.Variable(param2)

    def forward(self, x):
        b, v, l = x.size()
        out, _ = self.lstm(x, self.init_hidden(b))
        out = self.classifier(out)
        out = out.permute(0, 2, 1)
        return out
