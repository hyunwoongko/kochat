from abc import ABCMeta, abstractmethod

from backend.decorators import loss


@loss
class BaseLoss(metaclass=ABCMeta):

    @abstractmethod
    def compute_loss(self, label, logits, feats, mask=None):
        raise NotImplementedError

    def step(self, total_loss, optimizers):
        for opt in optimizers: opt.zero_grad()
        total_loss.backward()
        for opt in optimizers: opt.step()
        return total_loss
