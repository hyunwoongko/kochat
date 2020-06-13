"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import torch


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vector_size = 64  # word vector size
    batch_size = 512  # batch size for training
    max_len = 8  # max length for pad sequencing

    # path configs
    root_path = "/home/gusdnd852/Github/chatbot/"
    raw_datapath = root_path + "data/raw/"
    intent_datapath = root_path + "data/intent_data.csv"
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
    intent_intra_lr = 1e-4  # learning rate for intra loss
    intent_inter_lr = 1e-2  # learning rate for inter loss
    intent_weight_decay = 1e-4  # weight decay for intent training
    intent_epochs = 2000  # num of epoch for intent training
    intent_classes = 4  # num of intent class
    intent_ratio = 0.8  # intent train per test ratio
    intent_log_precision = 4  # floating point precision for logging
    last_dim = 256
    margin = 5.0
