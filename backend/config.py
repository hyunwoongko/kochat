"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
import sklearn
import torch
from sklearn import neighbors

BACKEND = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'root_dir': "/home/gusdnd852/Github/chatbot/backend/",  # 백엔드 루트경로
    'vector_size': 64,  # 단어 벡터 사이즈
    'batch_size': 256,  # 미니배치 사이즈
    'max_len': 8,  # 문장의 최대 길이 (패드 시퀀싱)
}

DATA = {
    'data_ratio': 0.8,  # 학습/검증 데이터 비율
    'raw_data_dir': BACKEND['root_dir'] + "data/raw/",  # 원본 데이터 파일 경로
    'intent_data_dir': BACKEND['root_dir'] + "data/intent_data.csv",  # 생성된 인텐트 데이터 파일 경로
    'entity_data_dir': BACKEND['root_dir'] + "data/entity_data.csv",  # 생성된 엔티티 데이터 파일 경로
    'ood_data_dir': BACKEND['root_dir'] + "data/ood/",  # 리트리벌 테스트를 위한 out of distribution 데이터셋

    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'TRAVEL'],
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}

PROC = {
    'logging_precision': 4,  # 로깅시 반올림 n번째에서 반올림

}

MODEL = {
    'model_dir': BACKEND['root_dir'] + "saved/",  # 모델 파일/시각화 자료 저장경로
}

LOSS = {
    'center_factor': 0.05,  # Center Loss의 weighting 비율
    'coco_alpha': 6.25,  # COCO loss의 alpha 값
    'cosface_s': 7.00,  # Cosface의 s값 (x^T dot W를 cos형식으로 바꿀 때 norm(||x||))
    'cosface_m': 0.25,  # Cosface의 m값 (Cosface의 마진)
    'gaussian_mixture_factor': 0.1,  # Gaussian Mixture Loss의 weighting 비율
    'gaussian_mixture_alpha': 0.00,  # Gaussian Mixture Loss의 alpha 값
}

GENSIM = {
    'window_size': 2,  # 임베딩 학습시 사용되는 윈도우 사이즈
    'workers': 8,  # 학습시 사용되는 쓰레드 워커 갯수
    'min_count': 2,  # 데이터에서 min count보다 많이 등장해야 단어로 인지
    'sg': 1,  # 0 : CBOW = 1 / SkipGram = 2
    'iter': 1500  # 임베딩 학습 횟수
}

INTENT = {
    'model_lr': 1e-4,  # 인텐트 학습시 사용되는 모델의 러닝레이트
    'loss_lr': 1e-2,  # 리트리벌 학습시 로스의 센터를 움직이는 러닝레이트
    'weight_decay': 1e-4,  # 인텐트 학습시 사용되는 가중치 감쇠 정도
    'epochs': 500,  # 인텐트 학습 횟수
    'd_model': 512,  # 인텐트 모델의 차원
    'd_loss': 128,  # 인텐트 로스의 차원 (시각화차원, 높을수록 ood 디텍션이 정확해지지만 느려집니다.)
    'layers': 13,  # 인텐트 모델의 레이어(층)의 수

    'visualization_epoch': 10,  # 시각화 빈도 (애폭마다 시각화 수행)
    'fallback_detction_threshold': -1,  # mean 혹은 min 선택시 임계값
    'fallback_detction_criteria' : 'auto', # [auto, min, mean]
    # auto를 쓰려면 ood dataset을 함께 넣어줘야합니다.

    'dist_param': {  # KNN 학습시 사용하는 그리드 서치 파라미터
        'n_neighbors': list(range(1, 50)),  # K값 범위 설정
        'weights': ["uniform", "distance"],  # 거리 가중치 부과 여부
        'p': [1, 2],  # 맨하튼 디스턴스 vs 유클리디언 디스턴스
        'algorithm': ['ball_tree', 'kd_tree']  # 탐색 기법
    },

    'lr_scheduler_factor': 0.75,  # 러닝레이트 스케줄러 감소율
    'lr_scheduler_patience': 10,  # 러닝레이트 스케줄러 감소 에폭
    'lr_scheduler_min_lr': 1e-12,  # 최소 러닝레이트
    'lr_scheduler_warm_up': 100  # 러닝레이트 감소 시작시점
}

ENTITY = {
    'model_lr': 1e-4,  # 엔티티 학습시 사용되는 모델의 러닝레이트
    'weight_decay': 1e-4,  # 엔티티 학습시 사용되는 가중치 감쇠 정도
    'epochs': 500,  # 엔티티 학습 횟수
    'd_model': 256,  ## 엔티티 모델의 차원
    'layers': 3,  # 엔티티 모델의 레이어(층)의 수

    'lr_scheduler_factor': 0.75,  # 러닝레이트 스케줄러 감소율
    'lr_scheduler_patience': 10,  # 러닝레이트 스케줄러 감소 에폭
    'lr_scheduler_min_lr': 1e-12,  # 최소 러닝레이트
    'lr_scheduler_warm_up': 100  # 러닝레이트 감소 시작시점
}
