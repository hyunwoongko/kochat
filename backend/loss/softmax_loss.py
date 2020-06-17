from torch import nn

from backend.decorators import loss


@loss
class SoftmaxLoss(nn.CrossEntropyLoss):

    def step(self, logits, feats, label, opts):
        total_loss = self(logits, label)
        for opt in opts: opt.zero_grad()
        total_loss.backward()
        for opt in opts: opt.step()
        return total_loss
