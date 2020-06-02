"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn
from config import Config


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = Config()
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
