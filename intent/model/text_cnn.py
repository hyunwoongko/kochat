"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from config import Config


class Conv(nn.Module):

    def __init__(self, _in, _out, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=_in, out_channels=_out, kernel_size=kernel_size)
        self.norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = Config()
        self.conv = Conv(self.conf.vector_size, 512, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.out = nn.Linear(4096, self.conf.intent_classes)

    def forward_once(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x.squeeze())
        return x

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return x1, x2
