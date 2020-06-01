"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vector_size = 128  # word vector size
    batch_size = 128  # batch size for training
    max_len = 16  # max length for pad sequencing

    # path configs
    root_path = "C:\\Users\\User\\Github\\Chatbot Deeplearning\\"
    intent_datapath = root_path + "data\\total_intent.csv"
    embed_storepath = root_path + "models\\embed\\embed.model"
    intent_storepath = root_path + "models\\intent\\intent.pth"

    # embed configs
    emb_window = 4  # window size for embedding training
    emb_workers = 8  # num of thread workers for embedding training
    emb_min_count = 0  # removing min count word during embedding training
    emb_sg = 1  # 0 : cbow / 1 : skip gram
    emb_iter = 500  # num of iteration for embedding training

    # intent configs
    intent_lr = 1e-5  # learning rate for intent training
    intent_weight_decay = 1e-3  # weight decay for intent training
    intent_epochs = 100  # num of epoch for intent training
    intent_classes = 3  # num of intent class
    intent_ratio = 0.8  # intent train per test ratio
    intent_log_precision = 5  # floating point precision for logging
    intent_loss = nn.CrossEntropyLoss()  # loss function for intent training
