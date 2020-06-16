import torch
from torch import nn

"""
code reference :
https://github.com/YirongMao/softmax_variants
"""


class COCOLoss(nn.Module):

    def __init__(self, d_model, label_dict, alpha=6.25):
        super(COCOLoss, self).__init__()
        self.d_model = d_model
        self.classes = len(label_dict)
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(self.classes, d_model))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.alpha * nfeat
        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))
        return logits
