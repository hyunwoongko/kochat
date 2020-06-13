"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn

from config import Config


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = Config()
        self.hidden_size = 256
        self.layer = 2
        self.direction = 2  # bidirectional
        self.lstm = nn.LSTM(input_size=self.conf.vector_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.layer,
                            bidirectional=True if self.direction == 2 else False)

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(torch.randn(self.layer * self.direction, batch_size, self.hidden_size).cuda()),
                torch.autograd.Variable(torch.randn(self.layer * self.direction, batch_size, self.hidden_size).cuda()))

    def forward(self, x):
        b, v, l = x.size()
        x = x.permute(2, 0, 1)
        x, (h_s, c_s) = self.lstm(x, self.init_hidden(b))

        print(h_s.size())
        print(h_s[-1].size())
        return h_s[-1]
