import torch
from torch import nn
from torch.nn import functional as F

from _backend.decorators import intent
from _backend.loss.base.base_loss import BaseLoss

"""
code reference :
https://github.com/YirongMao/softmax_variants
"""


@intent
class COCOLoss(nn.Module, BaseLoss):

    def __init__(self, label_dict):
        super(COCOLoss, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.coco_alpha * nfeat
        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))
        return logits

    def compute_loss(self, label, logits, feats, mask=None):
        logits = self(feats)
        return F.cross_entropy(logits, label)
