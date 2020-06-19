"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

from torch import nn

from backend.decorators import intent, model
from backend.model.base.convolution import Convolution


@intent
@model
class IntentCNN(nn.Module):

    def __init__(self, label_dict):
        super(IntentCNN, self).__init__()
        self.label_dict = label_dict
        self.stem = Convolution(self.vector_size, self.d_model, kernel_size=1)
        self.hidden_layers = nn.Sequential(*[
            Convolution(self.d_model, self.d_model, kernel_size=1)
            for _ in range(self.layers)])

        # visualization
        self.ret_features = nn.Linear(self.d_model * self.max_len, self.d_loss)
        self.ret_logits = nn.Linear(self.d_loss, len(self.label_dict))
        self.clf_logits = nn.Linear(self.d_model * self.max_len, len(self.label_dict))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.hidden_layers(x)
        x = x.view(x.size(0), -1)
        return x
