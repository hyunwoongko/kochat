"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from _backend.decorators import intent
from _backend.proc.base.torch_processor import TorchProcessor
from _backend.proc.distance_estimator import DistanceEstimator
from _backend.proc.fallback_detector import FallbackDetector


@intent
class IntentClassifier(TorchProcessor):

    def __init__(self, model, loss, debug=True):
        super().__init__(model)
        self.label_dict = model.label_dict
        self.loss = loss.to(self.device)
        self.debug = debug

        if len(list(loss.parameters())) != 0:
            loss_opt = SGD(params=loss.parameters(), lr=self.loss_lr)
            self.optimizers.append(loss_opt)

        self.features, self.ood_data = {}, None
        self.fallback = FallbackDetector()
        self.dist_estimator = DistanceEstimator(self.label_dict)
