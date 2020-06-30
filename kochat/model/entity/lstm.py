import torch
from torch import nn, autograd
from torch import Tensor
from kochat.decorators import entity


@entity
class LSTM(nn.Module):

    def __init__(self, label_dict: dict, bidirectional: bool = True):
        """
        Entity Recognition을 위한 LSTM 모델 클래스입니다.

        :param label_dict: 라벨 딕셔너리
        :param bidirectional: Bidirectional 여부
        """

        super().__init__()
        self.label_dict = label_dict
        self.direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=self.vector_size,
                            hidden_size=self.d_model,
                            num_layers=self.layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.out = nn.Linear(self.d_model * self.direction, len(label_dict))

    def init_hidden(self, batch_size: int) -> autograd.Variable:
        param1 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        param2 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        return torch.autograd.Variable(param1), torch.autograd.Variable(param2)

    def forward(self, x: Tensor) -> Tensor:
        b, l, v = x.size()
        out, _ = self.lstm(x, self.init_hidden(b))
        logits = self.out(out)
        logits = logits.permute(0, 2, 1)
        return logits
