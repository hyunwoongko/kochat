import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from kochat.decorators import intent
from kochat.loss.base.base_loss import BaseLoss


@intent
class COCOLoss(BaseLoss):

    def __init__(self, label_dict: dict):
        """
        COCO Loss를 계산합니다.

        - paper reference : https://arxiv.org/pdf/1710.00870.pdf
        - code reference : https://github.com/YirongMao/softmax_variants

        :param label_dict: 라벨 딕셔너리
        """

        super(COCOLoss, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))

    def forward(self, feat: Tensor):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.coco_alpha * nfeat
        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))
        return logits

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
        """
        학습을 위한 total loss를 계산합니다.

        :param label: label
        :param logits: logits
        :param feats: feature
        :param mask: mask vector
        :return: total loss
        """

        logits = self(feats)
        return F.cross_entropy(logits, label)
