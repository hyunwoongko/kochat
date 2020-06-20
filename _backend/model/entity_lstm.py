import torch
from torch import nn

from _backend.decorators import entity


@entity
class EntityLSTM(nn.Module):

    def __init__(self, label_dict, bidirectional=True):
        super().__init__()
        self.label_dict = label_dict
        self.direction = 2 if bidirectional else 1  # bidirectional
        self.lstm = nn.LSTM(input_size=self.vector_size,
                            hidden_size=self.d_model,
                            num_layers=self.layers,
                            batch_first=True,
                            bidirectional=True if self.direction == 2 else False)

        self.out = nn.Linear(self.d_model * self.direction, len(label_dict))

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(torch.randn(self.layers * self.direction, batch_size, self.d_model).cuda()),
                torch.autograd.Variable(torch.randn(self.layers * self.direction, batch_size, self.d_model).cuda()))

    def forward(self, x):
        b, l, v = x.size()
        out, _ = self.lstm(x, self.init_hidden(b))
        logits = self.out(out)
        logits = logits.permute(0, 2, 1)
        return logits
