"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, vector_size, d_model, layers, classes):
        super().__init__()
        self.vector_size = vector_size
        self.d_model = d_model
        self.layers = layers
        self.classes = classes
        self.direction = 2  # bidirectional

        self.lstm = nn.LSTM(input_size=self.vector_size,
                            hidden_size=self.d_model,
                            num_layers=self.layers,
                            batch_first=True,
                            bidirectional=True if self.direction == 2 else False)

        self.out = nn.Linear(self.d_model * 2, self.classes)

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(torch.randn(self.layers * self.direction, batch_size, self.d_model).cuda()),
                torch.autograd.Variable(torch.randn(self.layers * self.direction, batch_size, self.d_model).cuda()))

    def forward(self, x):
        b, v, l = x.size()
        # [max_len, batch_size, vector_size]
        out, _ = self.lstm(x, self.init_hidden(b))
        out = self.out(out)
        out = out.permute(0, 2, 1)
        return out
