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
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Net(nn.Module):

    def __init__(self):
        """
        인텐트 리트리벌 모델입니다.
        현재 보유한 데이터셋이 매우 작기 때문에 깊게 설계하지 않고 가볍게 설계했습니다.
        Convolution 블록 1층만 태우고, 이 feature를 Linear 레이어에서 분류합니다.

        인퍼런스 시에는 classifier를 사용하지 않고 distance 기반으로 metric learning된
        feature를 사용하여 nearest neighbors 탐색하여 인텐트를 검색합니다.
        threshold를 줘서 일정 거리 안에 유효한 샘플들이 없다면 Fallback처리를 수행합니다.
        """

        super().__init__()
        self.conf = Config()
        self.dim = self.conf.intent_net_dim // self.conf.max_len
        self.validate(self.dim)
        self.feature = Conv(self.conf.vector_size, self.dim, kernel_size=1)
        self.classifier = nn.Linear(self.conf.intent_net_dim, self.conf.intent_classes)

    def validate(self, dim):
        valid_list = [2 ** i for i in range(100)]
        if dim not in valid_list:
            raise Exception("\n\n(net_dim // max_len) must be 2 to the 'n'th power"
                            "\nyours is ({1} // {2}) = {0}"
                            "\n{0} should be in [2, 4, 8, 16, 32, ... 2^n]"
                            .format(self.dim, self.conf.intent_net_dim, self.conf.max_len))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x
