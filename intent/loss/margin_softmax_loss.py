import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class MarginSoftmaxLoss(nn.CrossEntropyLoss):
    """
    Intra Class인 데이터 샘플들 사이를 더욱 멀어지게 만드는 마진 로스함수입니다.

    Code Refernce: https://github.com/tk1980/LargeMarginInSoftmax
    TODO : 앵귤러 마진 로스들 (sphereface, cosface, arcface 등 실험해보기)
    """

    def __init__(self, ignore_index=-100):
        super(MarginSoftmaxLoss, self).__init__(weight=None,
                                                ignore_index=ignore_index,
                                                reduction='mean')
        self.conf = Config()
        self.reg_lambda = self.conf.reg_lambda

    def forward(self, x, labels):
        """
        :param x: [batch_size, classes]
        :param labels: [batch_size]
        :return: Softmax Loss + lambda * Margin Loss
        """

        batch_size, classes = x.size()

        # 클래스 마스킹
        mask = torch.zeros_like(x, requires_grad=False)
        mask[range(batch_size), labels] = 1

        # Corss Entropy 로스를 미리 구해놓음
        loss = F.cross_entropy(x, labels, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = x - 1.e6 * mask
        margin_loss = 0.5 * ((F.softmax(X, dim=1) - 1.0 / (classes - 1)) *
                             F.log_softmax(X, dim=1) * (1.0 - mask)).sum(dim=1)

        return loss + self.reg_lambda * margin_loss.mean()
