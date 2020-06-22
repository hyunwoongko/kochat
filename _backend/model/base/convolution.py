from torch import nn


class Convolution(nn.Module):

    def __init__(self, _in, _out, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=_in,
                              out_channels=_out,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)

        self.norm = nn.BatchNorm1d(_out)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        _x = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        # residual connection
        return x + _x \
            if x.size() == _x.size() and self.residual \
            else x
