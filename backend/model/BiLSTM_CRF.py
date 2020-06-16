"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn

from backend.decorators import model


@model
class Model(nn.Module):

    def __init__(self, vector_size, d_model, layers, label_dict):
        super().__init__()
        self.vector_size = vector_size
        self.d_model = d_model
        self.layers = layers
        self.classes = len(label_dict)
        self.direction = 2  # bidirectional

        self.lstm = nn.LSTM(input_size=self.vector_size,
                            hidden_size=self.d_model,
                            num_layers=self.layers,
                            batch_first=True,
                            bidirectional=True if self.direction == 2 else False)

        self.retrieval = nn.Sequential(nn.Linear(d_model * 2, 2), nn.ReLU())
        self.classifier = nn.Linear(2, self.classes)

    def init_hidden(self, batch_size):
        param1 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        param2 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        return torch.autograd.Variable(param1), torch.autograd.Variable(param2)

    def forward(self, x):
        b, v, l = x.size()
        out, _ = self.lstm(x, self.init_hidden(b))
        out = self.out(out)
        out = out.permute(0, 2, 1)
        return out
