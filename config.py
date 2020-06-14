"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch

BASE = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': "/home/gusdnd852/Github/chatbot/",  # project root path
    'vector_size': 128,  # word vector size
    'batch_size': 256,  # batch size for training
    'max_len': 8,  # max length for pad sequencing
    'logging_precision': 4,  # floating point precision for logging
    'data_ratio': 0.8,  # train data / test data ratio
    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'PURPOSE'],
    'NER_tagging': ['B', 'E', 'I', 'S'],  # BEGIN, END, INSIDE, SINGLE
    'NER_outside': 'O',  # empty or none tagging (means NOTHING)
}

DATA = {

    'raw_data_dir': BASE['root_dir'] + "data/raw/",
    'intent_data_file': BASE['root_dir'] + "data/intent_data.csv",
    'entity_data_file': BASE['root_dir'] + "data/entity_data.csv",
}

MODEL = {
    'embed_dir': BASE['root_dir'] + "models/embed",
    'intent_dir': BASE['root_dir'] + "models/intent",
    'entity_dir': BASE['root_dir'] + "models/entity",

    'embed_processor_file': BASE['root_dir'] + "models/embed/fasttext",
    'intent_classifier_file': BASE['root_dir'] + "/models/intent/classifier.pth",
    'intent_retrieval_file': BASE['root_dir'] + "/models/intent/retrieval.pth",
    'entity_recognizer_file': BASE['root_dir'] + "/models/entity/recognizer.pth",
}

EMBEDDING = {
    'window_size': 4,  # window size for embedding training
    'workers': 8,  # num of thread workers for embedding training
    'min_count': 1,  # removing min count word during embedding training
    'sg': 1,  # 0 : cbow / 1 : skip gram
    'iter': 2500  # num of iteration for embedding training
}

INTENT = {
    'intra_lr': 1e-4,  # learning rate for intra loss
    'inter_lr': 1e-2,  # learning rate for inter loss
    'weight_decay': 1e-4,  # weight decay for intent training
    'epochs': 2000,  # num of epoch for intent training
    'd_model': 512,  # model dimension for intent training
    'layers': 3,  # number of hidden layer for intent training
    'intra_factor': 0.3,  # intra loss weighting factor
    'inter_factor': 0.3  # inter loss weighting factor
}

ENTITY = {
    'lr': 1e-4,  # learning rate for entity training
    'weight_decay': 1e-4,  # weight decay for entity training
    'epochs': 20,  # num of epoch for entity training
    'd_model': 128,  ## model dimension for entity training
    'layers': 1  # number of hidden layer for entity training
}
