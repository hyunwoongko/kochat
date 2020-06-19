import torch
from torch import nn
from torch.autograd import Variable

from backend.decorators import model


@model
class Masking(nn.Module):
    """
    어디서든 편하게 사용하는 마스킹 레이어
    """

    def forward(self, sequence_length):
        batch_size = sequence_length.size(0)
        seq_range = torch.range(0, self.max_len - 1).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, self.max_len)
        seq_range_expand = Variable(seq_range_expand).to(self.device)
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand)).to(self.device)
        return seq_range_expand < seq_length_expand
