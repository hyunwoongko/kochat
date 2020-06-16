"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""

from torch import nn
from backend.decorators import model
from backend.model.sub_layer.convolution import Convolution


@model
class Model(nn.Module):

    def __init__(self, vector_size, d_model, layers, label_dict):
        super(Model, self).__init__()
        self.classes = len(label_dict)
        self.stem = Convolution(vector_size, d_model, kernel_size=1)
        self.feature = nn.Sequential(*[
            Convolution(d_model, d_model, kernel_size=1)
            for _ in range(layers)
        ])

        self.retrieval = nn.Sequential(nn.Linear(d_model, 2), nn.ReLU())
        self.classifier = nn.Linear(2, self.classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x
