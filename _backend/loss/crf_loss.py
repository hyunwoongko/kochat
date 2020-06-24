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
        """
        학습을 위한 total loss를 계산합니다.

        :param label: label
        :param logits: logits
        :param feats: feature
        :param mask: mask vector
        :return: total loss
        """

        logits = logits.permute(0, 2, 1)
        log_likelihood = self.crf(logits, label, mask=mask, reduction='mean')
        return - log_likelihood  # nll loss

    def decode(self, logits, mask=None):
        """
        Viterbi Decoding의 구현체입니다.
        CRF 레이어의 출력을 prediction으로 변형합니다.

        :param logits: 모델의 출력 (로짓)
        :param mask: 마스킹 벡터
        :return: 모델의 예측 (prediction)
        """

        logits = logits.permute(0, 2, 1)
        return self.crf.decode(logits, mask)
