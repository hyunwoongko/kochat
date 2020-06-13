import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginSoftmaxLoss(nn.CrossEntropyLoss):

    def __init__(self, reg_lambda=0.3,
                 weight=None,
                 ignore_index=-100,
                 reduction='mean'):
        super(MarginSoftmaxLoss, self).__init__(weight=weight,
                                                ignore_index=ignore_index,
                                                reduction=reduction)
        self.reg_lambda = reg_lambda

    def forward(self, x, labels):
        batch_size, classes = x.size()
        mask = torch.zeros_like(x, requires_grad=False)
        mask[range(batch_size), labels] = 1

        loss = F.cross_entropy(x, labels, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = x - 1.e6 * mask  # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0 / (classes - 1)) *
                     F.log_softmax(X, dim=1) * (1.0 - mask)).sum(dim=1)

        reg = reg.mean()
        return loss + self.reg_lambda * reg
