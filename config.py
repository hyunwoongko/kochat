"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import torch


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vector_size = 64  # word vector size
    batch_size = 256  # batch size for training
    max_len = 8  # max length for pad sequencing

    # path configs
    root_path = "/home/gusdnd852/Github/chatbot/"
    intent_datapath = root_path + "data/total_intent.csv"
    embed_storepath = root_path + "models/embed"
    embed_storefile = embed_storepath + '/embed.model'
    intent_storepath = root_path + "models/intent"
    intent_storefile = intent_storepath + '/intent.pth'
    siamese_storefile = intent_storepath + '/siamese.pth'
    center_storefile = intent_storepath + '/center.pth'

    # embed configs
    emb_window = 4  # window size for embedding training
    emb_workers = 8  # num of thread workers for embedding training
    emb_min_count = 2  # removing min count word during embedding training
    emb_sg = 1  # 0 : cbow / 1 : skip gram
    emb_iter = 1000  # num of iteration for embedding training

    # intent configs
    intent_lr = 1e-5  # learning rate for intent training
    intent_weight_decay = 5e-3  # weight decay for intent training
    intent_epochs = 500  # num of epoch for intent training
    intent_classes = 4  # num of intent class
    intent_ratio = 0.8  # intent train per test ratio
    intent_log_precision = 4  # floating point precision for logging

    siamese_lr = 1e-5  # learning rate for intent training
    siamese_weight_decay = 5e-3  # weight decay for intent training
    siamese_epochs = 5000  # num of epoch for intent training
    siamese_classes = intent_classes  # num of intent class
    siamese_ratio = 0.8  # intent train per test ratio
    siamese_log_precision = 4  # floating point precision for logging

    center_epochs = 20000  # num of epoch for intent training
