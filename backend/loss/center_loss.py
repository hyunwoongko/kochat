import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn import functional as F

from backend.decorators import intent
from backend.loss.base.base_loss import BaseLoss

"""
code reference :
https://github.com/YirongMao/softmax_variants
"""


@intent
class CenterLoss(nn.Module, BaseLoss):
    def __init__(self, label_dict):
        super(CenterLoss, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))
        self.center_loss_function = CenterLossFunction.apply

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()

        if feat.size(1) != self.d_loss:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}"
                             .format(self.d_loss, feat.size(1)))

        return self.center_loss_function(feat, label, self.centers)

    def compute_loss(self, label, logits, feats, mask=None):
        nll_loss = F.cross_entropy(logits, label)
        center_loss = self(feats, label)
        return nll_loss + self.center_factor * center_loss


class CenterLossFunction(Function):

    @staticmethod
    def forward(ctx, feat, label, centers):
        ctx.save_for_backward(feat, label, centers)
        centers_pred = centers.index_select(0, label.long())
        return (feat - centers_pred).pow(2).sum(1).sum(0) / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        grad_feature = feature - centers.index_select(0, label.long())

        counts = torch.ones(centers.size(0))
        grad_centers = torch.zeros(centers.size())
        if feature.is_cuda:
            counts = counts.cuda()
            grad_centers = grad_centers.cuda()

        for i in range(feature.size(0)):
            j = int(label[i].data)
            counts[j] += 1
            grad_centers[j] += (centers.data[j] - feature.data[i])

        grad_centers = Variable(grad_centers / counts.view(-1, 1))
        return grad_feature * grad_output, None, grad_centers
