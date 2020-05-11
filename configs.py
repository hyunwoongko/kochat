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


class FastTextConfigs:
    window = 4
    workers = 8
    min_count = 0
    sg = 1
    iter = 500


class TransformerClassifierConfigs:
    d_model = 512
    ffn_hidden = 1024
    n_heads = 8
    drop_prob = 0.1
    n_layers = 6

    epochs = 250
    lr = 1e-5
    weight_decay = 0.01
