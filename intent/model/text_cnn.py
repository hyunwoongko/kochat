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

    def residual(self, x, _x):
        return x + _x if x.size() == _x.size() else x

    def forward(self, x):
        _x = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.residual(x, _x)
        x = self.relu(x)
        return x


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = Config()
        self.layer = nn.Sequential(
            Conv(self.conf.vector_size, 256, kernel_size=1),
            Conv(256, 256, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Conv(256, 512, kernel_size=1),
            Conv(512, 512, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.out = nn.Linear(2048, 2)

    def forward_once(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.out(x.squeeze())
        return x

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return x1, x2
