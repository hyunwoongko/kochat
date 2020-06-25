"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from torch import Tensor
from _backend.decorators import intent
from _backend.model.base.convolution import Convolution


@intent
class IntentCNN(nn.Module):

    def __init__(self, label_dict: dict, residual: bool = True):
        """
        Intent Classification을 위한 CNN 클래스입니다.

        :param label_dict: 라벨 딕셔너리
        :param residual: skip connection 여부
        """

        super(IntentCNN, self).__init__()
        self.label_dict = label_dict
        self.stem = Convolution(self.vector_size, self.d_model, kernel_size=1, residual=residual)
        self.hidden_layers = nn.Sequential(*[
            Convolution(self.d_model, self.d_model, kernel_size=1, residual=residual)
            for _ in range(self.layers)])

        # ret features, logits => retrieval시 사용
        # clf logits => softmax classification시 사용
        self.ret_features = nn.Linear(self.d_model * self.max_len, self.d_loss)
        self.ret_logits = nn.Linear(self.d_loss, len(self.label_dict))
        self.clf_logits = nn.Linear(self.d_model * self.max_len, len(self.label_dict))

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.hidden_layers(x)
        x = x.view(x.size(0), -1)
        return x
