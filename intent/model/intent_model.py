"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import math

import torch
from torch import nn
from config import Config


class Conv(nn.Module):
    """
    기본적인 Conv - BN - Relu 블록입니다.
    """

    def __init__(self, _in, _out, kernel_size):
        super().__init__()
        self.softmax = nn.Softmax()
        self.conv = nn.Conv1d(in_channels=_in,
                              out_channels=_out,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        _x = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x + _x if x.size() == _x.size() else x
        # residual connection


class Net(nn.Module):

    def __init__(self):
        """
        인텐트 리트리벌 CNN 모델입니다.
        """

        super().__init__()
        self.conf = Config()
        self.dim = self.conf.intent_net_dim // self.conf.max_len
        self.validate(self.dim)
        self.stem = Conv(self.conf.vector_size, self.dim, kernel_size=1)
        self.feature = nn.Sequential(*[Conv(self.dim, self.dim, kernel_size=1)
                                       for _ in range(self.conf.intent_net_layers)])

        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(self.conf.intent_net_dim, self.conf.intent_classes))

    def validate(self, dim):
        valid_list = [2 ** i for i in range(100)]
        if dim not in valid_list:
            raise Exception("\n\n(net_dim // max_len) must be 2 to the 'n'th power"
                            "\nyours is ({1} // {2}) = {0}"
                            "\n{0} should be in [2, 4, 8, 16, 32, ... 2^n]"
                            .format(self.dim, self.conf.intent_net_dim, self.conf.max_len))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x
