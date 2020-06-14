"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

from torch import nn

from base.model_managers.model_manager import Intent


class Conv(nn.Module, Intent):
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
        # residual connection


class Model(nn.Module):

    def __init__(self, vector_size, max_len, d_model, layers, classes):
        """
        인텐트 리트리벌 CNN 모델입니다.
        """

        super(Model, self).__init__()
        self.dim = d_model // max_len
        self.validate(self.dim)
        self.stem = Conv(vector_size, self.dim, kernel_size=1)

        self.feature = nn.Sequential(*[Conv(self.dim, self.dim, kernel_size=1)
                                       for _ in range(layers)])

        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(d_model, classes))

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
