"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""

import torch

BACKEND = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': "/home/gusdnd852/Github/chatbot/backend/",  # 백엔드 루트경로
    'vector_size': 64,  # 단어 벡터 사이즈
    'batch_size': 256,  # 미니배치 사이즈
    'max_len': 8,  # 문장의 최대 길이 (패드 시퀀싱)
    'logging_precision': 4,  # 로깅시 반올림 n번째에서 반올림
    'data_ratio': 0.8,  # 학습/검증 데이터 비율
    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'TRAVEL'],
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}

DATA = {
    'raw_data_dir': BACKEND['root_dir'] + "data/raw/", # 원본 데이터 파일 경로
    'intent_data_dir': BACKEND['root_dir'] + "data/intent_data.csv", # 생성된 인텐트 데이터 파일 경로
    'entity_data_dir': BACKEND['root_dir'] + "data/entity_data.csv", # 생성된 엔티티 데이터 파일 경로
}

PROC = {
    'logs_dir': BACKEND['root_dir'] + "saved/logs/", # 로깅/시각과 저장 경로
}

MODEL = {
    'model_dir': BACKEND['root_dir'] + "saved/models/", # 모델파일 저장경로 (.pth)
}

LOSS = {
    'center_factor': 0.01,  # Center Loss의 weighting 비율
    'coco_alpha': 6.25,  # COCO loss의 alpha 값
    'cosface_s': 7.00,  # Cosface의 s값 (x^T dot W를 cos형식으로 바꿀 때 norm(||x||))
    'cosface_m': 0.2,  # Cosface의 m값 (Cosface의 마진)
    'gaussian_mixture_factor': 0.1,  # Gaussian Mixture Loss의 weighting 비율
    'gaussian_mixture_alpha': 0.00,  # Gaussian Mixture Loss의 alpha 값
}

GENSIM = {
    'window_size': 4,  # 임베딩 학습시 사용되는 윈도우 사이즈
    'workers': 8,  # 학습시 사용되는 쓰레드 워커 갯수
    'min_count': 1,  # 데이터에서 min count보다 많이 등장해야 단어로 인지
    'sg': 1,  # 0 : CBOW = 1 / SkipGram = 2
    'iter': 1500  # 임베딩 학습 횟수
}

INTENT = {
    'model_lr': 1e-4,  # 인텐트 학습시 사용되는 모델의 러닝레이트
    'loss_lr': 1e-2,  # 리트리벌 학습시 로스의 센터를 움직이는 러닝레이트
    'weight_decay': 1e-4,  # 인텐트 학습시 사용되는 가중치 감쇠 정도
    'epochs': 2000,  # 인텐트 학습 횟수
    'd_model': 256,  # 인텐트 모델의 차원
    'd_loss': 2,  # 인텐트 로스의 차원 (피쳐 스페이스 시각화 차원 = 2D)
    'layers': 3,  # 인텐트 모델의 레이어(층)의 수
}

ENTITY = {
    'model_lr': 1e-4,  # 엔티티 학습시 사용되는 모델의 러닝레이트
    'weight_decay': 1e-4,  # 엔티티 학습시 사용되는 가중치 감쇠 정도
    'epochs': 500,  # 엔티티 학습 횟수
    'd_model': 256,  ## 엔티티 모델의 차원
    'layers': 3  # 엔티티 모델의 레이어(층)의 수
}
