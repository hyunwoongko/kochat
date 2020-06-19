import torch
from torch import nn

from backend.loss.base.base_loss import BaseLoss
from torch.nn import functional as F


class SoftmaxLoss(nn.CrossEntropyLoss, BaseLoss):

    def __init__(self, label_dict):
        super(SoftmaxLoss, self).__init__()
        self.classes = len(label_dict)

    def compute_loss(self, label, logits, feats, mask=None):
        if mask is None:
            return self(logits, label)
        else:
            logits = logits.permute(0, 2, 1)
            logits_flat = logits.view(-1, logits.size(-1))
            log_probs_flat = F.log_softmax(logits_flat)
            target_flat = label.view(-1, 1)
            losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
            losses = losses_flat.view(mask.size())
            losses = losses * mask.float()
            return losses.mean()
