from torchcrf import CRF
from torch import nn

from _backend.decorators import entity
from _backend.loss.base.base_loss import BaseLoss


@entity
class CRFLoss(nn.Module, BaseLoss):

    def __init__(self, label_dict):
        super().__init__()
        self.label_dict = label_dict
        self.crf = CRF(len(label_dict), batch_first=True)

    def compute_loss(self, label, logits, feats, mask=None):
        logits = logits.permute(0, 2, 1)
        log_likelihood = self.crf(logits, label, mask=mask, reduction='mean')
        return - log_likelihood

    def decode(self, logits, mask=None):
        logits = logits.permute(0, 2, 1)
        return self.crf.decode(logits, mask)
