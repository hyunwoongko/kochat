"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import math

from configs import TransformerClassifierConfigs, GlobalConfigs
import torch
from torch import nn


class PostionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):
        super(PostionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :]


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, e=1e-12):
        batch_size, head, length, d_tensor = k.size()
        k_t = k.view(batch_size, head, d_tensor, length)
        score = (q @ k_t) / math.sqrt(d_tensor)
        score = self.softmax(score)
        return score @ v


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, max_len):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Conv1d(d_model, max_len, kernel_size=1)
        self.w_k = nn.Conv1d(d_model, max_len, kernel_size=1)
        self.w_v = nn.Conv1d(d_model, max_len, kernel_size=1)
        self.out = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.project(q), self.project(k), self.project(v)
        out = self.attention(q, k, v)
        out = self.concat(out)
        out = self.out(out)
        return out

    def project(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.view(batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, max_len):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, max_len=max_len)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        _x = x
        x = self.attention(x, x, x)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


class TransformerClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = TransformerClassifierConfigs()
        self.glb_conf = GlobalConfigs()
        self.pos_emb = PostionalEncoding(d_model=self.conf.d_model,
                                         max_len=self.glb_conf.max_len,
                                         device=self.glb_conf.device)

        self.encoder = nn.ModuleList([EncoderLayer(d_model=self.conf.d_model,
                                                   ffn_hidden=self.conf.ffn_hidden,
                                                   n_head=self.conf.n_heads,
                                                   drop_prob=self.conf.drop_prob,
                                                   max_len=self.glb_conf.max_len)
                                      for _ in range(self.conf.n_layers)])

    def forward(self, x):
        print(x.size())
        x = self.pos_emb(x)
        print(x.size())

        for layer in self.encoder:
            x = layer(x)

        x = self.out(x)
        return x
