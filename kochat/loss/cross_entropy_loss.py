import torch
from torch.nn import functional as F
from torch import Tensor
from torch import nn
from kochat.loss.base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):

    def __init__(self, label_dict: dict):
        """
        cross entropy loss를 계산합니다.

        :param label_dict: 라벨 딕셔너리
        """

        super(CrossEntropyLoss, self).__init__()
        self.classes = len(label_dict)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target)

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor = None, mask: nn.Module = None) -> Tensor:
        """
        학습을 위한 total loss를 계산합니다.

        :param label: label
        :param logits: logits
        :param feats: feature
        :param mask: mask vector
        :return: total loss
        """

        if mask is None:
            # 마스크 없는 경우 torch의 cross entropy 이용
            return self(logits, label)

        else:
            # 마스크 있는 경우 마스크를 처리하고 cross entropy 계산
            logits = logits.permute(0, 2, 1)
            logits_flat = logits.view(-1, logits.size(-1))
            log_probs_flat = F.log_softmax(logits_flat, dim=1)
            target_flat = label.view(-1, 1)
            losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
            losses = losses_flat.view(mask.size())
            losses = losses * mask.float()
            return losses.mean()
