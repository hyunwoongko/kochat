"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn
from config import Config


class Conv(nn.Module):

    def __init__(self, _in, _out, kernel_size):
        super().__init__()
        self.softmax = nn.Softmax()
        self.conv = nn.Conv1d(in_channels=_in, out_channels=_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        _x = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x + _x if x.size() == _x.size() else x


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = Config()
        self.layer = nn.Sequential(
            Conv(self.conf.vector_size, 256, kernel_size=1),
            Conv(256, 256, kernel_size=1),
            Conv(256, 256, kernel_size=1),
            Conv(256, 256, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.conf.intent_classes))

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        return x
