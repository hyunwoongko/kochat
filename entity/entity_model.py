"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn
import torch.nn.utils.rnn as R
from config import Config


class Net(nn.Module):

    def __init__(self, label_size):
        super().__init__()
        self.conf = Config()
        self.hidden_size = self.conf.entity_net_dim
        self.label_size = label_size
        self.layer = self.conf.entity_net_layers
        self.direction = 2  # bidirectional
        self.lstm = nn.LSTM(input_size=self.conf.vector_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.layer,
                            batch_first=True,
                            bidirectional=True if self.direction == 2 else False)

        self.out = nn.Linear(self.hidden_size * 2, self.label_size)

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(torch.randn(self.layer * self.direction, batch_size, self.hidden_size).cuda()),
                torch.autograd.Variable(torch.randn(self.layer * self.direction, batch_size, self.hidden_size).cuda()))

    def forward(self, x):
        b, v, l = x.size()
        # [max_len, batch_size, vector_size]
        out, _ = self.lstm(x, self.init_hidden(b))
        out = self.out(out)
        out = out.permute(0, 2, 1)
        return out
