from torch import nn

from backend.decorators import loss
from backend.loss.base_loss import BaseLoss


@loss
class SoftmaxLoss(nn.CrossEntropyLoss, BaseLoss):

    def __init__(self, label_dict):
        super(SoftmaxLoss, self).__init__()
        self.classes = len(label_dict)

    def compute_loss(self, logits, feats, label):
        return self(logits, label)
