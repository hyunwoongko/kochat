"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch

BACKEND = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': "/home/gusdnd852/Github/chatbot/backend/",  # backend root path
    'vector_size': 64,  # word vector size
    'batch_size': 256,  # batch size for training
    'max_len': 8,  # max length for pad sequencing
    'logging_precision': 4,  # floating point precision for logging
    'data_ratio': 0.8,  # train data / test data ratio
    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'TRAVEL'],
    'NER_tagging': ['B', 'E', 'I', 'S'],  # BEGIN, END, INSIDE, SINGLE
    'NER_outside': 'O',  # empty or none tagging (means NOTHING)
}

DATA = {
    'raw_data_dir': BACKEND['root_dir'] + "data/raw/",
    'intent_data_dir': BACKEND['root_dir'] + "data/intent_data.csv",
    'entity_data_dir': BACKEND['root_dir'] + "data/entity_data.csv",
}

PROC = {
    'logs_dir': BACKEND['root_dir'] + "saved/logs/",
}

MODEL = {
    'model_dir': BACKEND['root_dir'] + "saved/models/",
}

LOSS = {
    'loss_factor': 0.3  # power of additional loss function
}

GENSIM = {
    'window_size': 4,  # window size for embedding training
    'workers': 8,  # num of thread workers for embedding training
    'min_count': 1,  # removing min count word during embedding training
    'sg': 1,  # 0 : cbow / 1 : skip gram
    'iter': 1500  # num of iteration for embedding training
}

INTENT = {
    'intra_lr': 1e-4,  # learning rate for intra loss
    'inter_lr': 1e-2,  # learning rate for inter loss
    'weight_decay': 1e-4,  # weight decay for intent training
    'epochs': 100,  # num of epoch for intent training
    'd_model': 128,  # model dimension for intent training
    'layers': 0,  # number of hidden layer for intent training
}

ENTITY = {
    'lr': 1e-4,  # learning rate for entity training
    'weight_decay': 1e-4,  # weight decay for entity training
    'epochs': 50,  # num of epoch for entity training
    'd_model': 128,  ## model dimension for entity training
    'layers': 1  # number of hidden layer for entity training
}
