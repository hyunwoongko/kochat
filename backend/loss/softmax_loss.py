from torch import nn

from backend.decorators import loss
from backend.loss.base_loss import BaseLoss


@loss
class SoftmaxLoss(nn.CrossEntropyLoss, BaseLoss):

    def compute_loss(self, logits, feats, label):
        return self(logits, label)
