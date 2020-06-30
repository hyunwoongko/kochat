from abc import abstractmethod

from torch import Tensor
from torch import nn

from kochat.decorators import loss


@loss
class BaseLoss(nn.Module):

    def __init__(self):
        """
        모든 Loss 함수들의 부모 클래스가 되는 클래스입니다.
        두개의 메소드 (compute_loss, step)을 가지고 있습니다.
        """

        super().__init__()

    @abstractmethod
    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
        """
        전체 Loss를 계산합니다. Loss함수들이 기본적으로 가지고 있는 forward와 다른점은
        예를 들자면 Center Loss의 경우 Softmax Loss + Center Loss가 전체 Loss가 됩니다.
        이 때, forward는 Center Loss의 정의대로 자기 자신의 Loss만을 설명하고 리턴하는 영역이라면,
        compute_loss는 자기자신의 loss와 softmax loss를 더한, 전체적인 loss를 의미합니다.

        :param label: 해당 배치의 라벨들
        :param logits: softmax classifier를 거친 logits
        :param feats: softmax classifier를 거치지 않은 features
        :param mask: 마스킹 추가를 원하면 여기에 마스킹 텐서를 넣습니다.
        :return: total loss를 반환합니다.
        """

        raise NotImplementedError
