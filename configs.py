"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import torch


class GlobalConfigs:
    root_path = "C:\\Users\\User\\Github\\Chatbot\\"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    intent_path = root_path + "data\\total_intent.csv"
    max_len = 16
    vector_size = 64
    batch_size = 128
    record_step = 1
    classes = 3


class FastTextConfigs:
    window = 4
    workers = 8
    min_count = 0
    sg = 1
    iter = 500


class ResNetConfigs:
    lr = 1e-5
    weight_decay = 1e-3
    epochs = 500
