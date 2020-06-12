from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        pos, neg, size = 0, 0, 0
        for i, x in enumerate(zip(target, distances)):
            if x[0].item() == 1:
                pos += x[1].item()
            else:
                neg += x[1].item()
            size += 1

        return losses.mean() if size_average else losses.sum()
