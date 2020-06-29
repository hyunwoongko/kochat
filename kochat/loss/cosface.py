import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import Tensor
from kochat.decorators import intent
from kochat.loss.base_loss import BaseLoss


@intent
class CosFace(BaseLoss):

    def __init__(self, label_dict: dict):
        """
        Cosface (=LMCL) Loss를 계산합니다

        - paper reference : https://arxiv.org/abs/1801.09414
        - code reference : https://github.com/YirongMao/softmax_variants

        :param label_dict: 라벨 딕셔너리
        """

        super(CosFace, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))

    def forward(self, feat: Tensor, label: Tensor) -> Tensor:
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.cosface_m)
        margin_logits = self.cosface_s * (logits - y_onehot)
        return margin_logits

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
        """
        학습을 위한 total loss를 계산합니다.

        :param label: label
        :param logits: logits
        :param feats: feature
        :param mask: mask vector
        :return: total loss
        """

        mlogits = self(feats, label)
        return F.cross_entropy(mlogits, label)
