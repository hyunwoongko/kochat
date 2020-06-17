import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from backend.decorators import intent, loss
from backend.loss.base_loss import BaseLoss

"""
code reference :
https://github.com/YirongMao/softmax_variants
"""


@intent
@loss
class LargeMarginGaussianMixture(nn.Module, BaseLoss):

    def __init__(self, label_dict):
        super(LargeMarginGaussianMixture, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))
        self.log_covs = nn.Parameter(torch.zeros(self.classes, self.d_loss))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)

        covs = torch.exp(log_covs)
        tcovs = covs.repeat(batch_size, 1, 1)
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.gaussian_mixture_alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1)
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5 * (tslog_covs + margin_dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5 * torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0 / batch_size) * (cdist + reg)
        return margin_logits, likelihood

    def compute_loss(self, logits, feats, label):
        mlogits, likelihood = self(feats)
        logits = F.cross_entropy(mlogits, label)
        return logits + self.gaussian_mixture_factor * likelihood
