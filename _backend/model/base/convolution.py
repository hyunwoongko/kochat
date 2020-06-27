import torch
from torch import nn
from torch import Tensor


class Convolution(nn.Module):

    def __init__(self, _in: int, _out: int, kernel_size: int, residual: bool):
        """
        기본적인 Convolution - BN - Relu 블럭입니다.

        :param _in: 입력 채널 사이즈
        :param _out: 출력 채널 사이즈
        :param kernel_size: 커널 사이즈
        :param residual: skip connection 여부
        """

        super().__init__()
        self.conv = nn.Conv1d(in_channels=_in,
                              out_channels=_out,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)

        self.norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x: Tensor) -> Tensor:
        _x = x
        x = self.conv(x)  # convolution
        x = self.norm(x)  # batch normalization
        x = self.relu(x)  # relu activation

        # residual connection
        return x + _x \
            if x.size() == _x.size() and self.residual \
            else x
