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
    'data_ratio': 0.8,  # 학습/검증 데이터 비율
    'logging_precision': 4,  # 로깅시 반올림 n번째에서 반올림
    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'TRAVEL'],
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}

DATA = {
    'raw_data_dir': BACKEND['root_dir'] + "data/raw/",  # 원본 데이터 파일 경로
    'intent_data_dir': BACKEND['root_dir'] + "data/intent_data.csv",  # 생성된 인텐트 데이터 파일 경로
    'entity_data_dir': BACKEND['root_dir'] + "data/entity_data.csv",  # 생성된 엔티티 데이터 파일 경로
    'ood_data_dir': BACKEND['root_dir'] + "data/ood/",  # 리트리벌 테스트를 위한 out of distribution 데이터셋
}

MODEL = {
    'model_dir': BACKEND['root_dir'] + "saved/",  # 모델 파일/시각화 자료 저장경로 (.pth)
}

LOSS = {
    'center_factor': 0.01,  # Center Loss의 weighting 비율
    'coco_alpha': 6.25,  # COCO loss의 alpha 값
    'cosface_s': 3.00,  # Cosface의 s값 (x^T dot W를 cos형식으로 바꿀 때 norm(||x||))
    'cosface_m': 0.1,  # Cosface의 m값 (Cosface의 마진)
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
    'epochs': 1024,  # 인텐트 학습 횟수
    'd_model': 512,  # 인텐트 모델의 차원
    'd_loss': 2,  # 인텐트 로스의 차원 (피쳐 스페이스 시각화 차원 = 2D / 3D / ND)
    'layers': 3,  # 인텐트 모델의 레이어(층)의 수
    'memory_dir': BACKEND['root_dir'] + "saved/memory/",  # 리트리벌 학습된 샘플들의 피쳐를 저장하는 공간

    # 시각화 기능을 끄려면 epoch을 위의 학습 epoch과 동일하게 설정하세요.
    'visualization_epoch': 10,  # 시각화 빈도 (d_loss > 3 이상에서는 차원축소가 들어가니 높게 설정하세요)
    'knn_param_grid': {  # KNN 학습시 사용하는 그리드 서치 파라미터
        'n_neighbors': list(range(1, 50)),
        'weights': ["uniform", "distance"],
        'p': [1, 2],  # 맨하튼 vs 유클리디언,
        'algorithm': ['ball_tree', 'kd_tree']
    }
}

ENTITY = {
    'model_lr': 1e-4,  # 엔티티 학습시 사용되는 모델의 러닝레이트
    'weight_decay': 1e-4,  # 엔티티 학습시 사용되는 가중치 감쇠 정도
    'epochs': 500,  # 엔티티 학습 횟수
    'd_model': 256,  ## 엔티티 모델의 차원
    'layers': 3  # 엔티티 모델의 레이어(층)의 수
}
