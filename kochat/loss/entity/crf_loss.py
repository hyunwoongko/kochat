from torch import Tensor
from torch import nn
from torchcrf import CRF

from kochat.decorators import entity
from kochat.loss.base.base_loss import BaseLoss


@entity
class CRFLoss(BaseLoss):

    def __init__(self, label_dict: dict):
        """
        Conditional Random Field를 계산하여 Loss 함수로 활용합니다.

        :param label_dict: 라벨 딕셔너리
        """

        super().__init__()
        self.classes = len(label_dict)
        self.crf = CRF(self.classes, batch_first=True)

    def decode(self, logits: Tensor, mask: nn.Module = None) -> list:
        """
        Viterbi Decoding의 구현체입니다.
        CRF 레이어의 출력을 prediction으로 변형합니다.

        :param logits: 모델의 출력 (로짓)
        :param mask: 마스킹 벡터
        :return: 모델의 예측 (prediction)
        """

        logits = logits.permute(0, 2, 1)
        return self.crf.decode(logits, mask)

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
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
