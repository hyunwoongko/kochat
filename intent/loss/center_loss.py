"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn

from config import Config


class CenterLoss(nn.Module):
    """
    Inter Class인 데이터 샘플들이 임베딩 스페이스에서 더욱 밀집하게 만들어주는 로스함수 입니다.
    기존 Center Loss는 해당 클래스의 중심 부분으로 밀집시키지만,
    이 구현에서는 임의의 점에 밀집시키고, Backpropation으로 적절한 위치에 이동시킵니다.
    (Intra Class 마진이 목표가 아니라 Inter Class 밀집이 목표이기 때문에 별 무리 없음)

    Code Refernce: https://github.com/KaiyangZhou/pytorch-center-loss
    """

    def __init__(self):
        super(CenterLoss, self).__init__()
        self.conf = Config()
        self.num_classes = self.conf.intent_classes
        self.reg_gamma = self.conf.reg_gamma
        self.feat_dim = self.conf.intent_net_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        # 구현의 편의를 위해 실제 중심이 아닌 임의의 점으로 모이게 만듬
        # 임의의 점이 Backpropagation에 의해 이동하기 때문에 동일한 효과

    def forward(self, x, labels):
        """
        Center Loss = 1/2 ∑(||x - c||_2)^2
                    = 1/2 ∑(x^2 - 2xc + c^2)

        Total Loss = Softmax Loss + gamma * Center Loss
        (gamma = regulation factor)

        :param x: [batch_size, dims]
        :param labels: [batch_size]
        :return: Center Loss를 리턴합니다.
        """

        batch_size = x.size(0)

        # distmat = x^2 + c^2
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # distmat = x^2 + c^2 - 2xc
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        # 클래스 마스킹
        classes = torch.arange(self.num_classes).long().cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # 마스킹 적용 후, gamma에 곱해서 리턴
        dist = 0.5 * distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss * self.reg_gamma
