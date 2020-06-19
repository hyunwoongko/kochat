from torch import nn
from torchcrf import CRF

from backend.decorators import entity
from backend.loss.base.base_loss import BaseLoss


@entity
class CRFLoss(nn.Module, BaseLoss):

    def __init__(self, label_dict):
        super().__init__()
        self.label_dict = label_dict
        self.crf = CRF(len(label_dict), batch_first=True)

    def compute_loss(self, label, logits, feats, mask=None):
        return -self.crf(logits, label, reduction='mean', mask=mask)
