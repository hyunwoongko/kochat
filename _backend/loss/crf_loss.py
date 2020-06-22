from torchcrf import CRF

from _backend.decorators import entity
from _backend.loss.base.base_loss import BaseLoss


@entity
class CRFLoss(BaseLoss):

    def __init__(self, label_dict):
        """
        Conditional Random Field를 계산하여 Loss 함수로 활용합니다.

        :param label_dict: 라벨 딕셔너리
        """

        super().__init__()
        self.label_dict = label_dict
        self.crf = CRF(len(label_dict), batch_first=True)

    def compute_loss(self, label, logits, feats, mask=None):
        logits = logits.permute(0, 2, 1)
        log_likelihood = self.crf(logits, label, mask=mask, reduction='mean')
        return - log_likelihood  # nll loss

    def decode(self, logits, mask=None):
        logits = logits.permute(0, 2, 1)
        return self.crf.decode(logits, mask)
