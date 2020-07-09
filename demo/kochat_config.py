import os
import platform
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

root_dir = os.path.abspath(os.curdir)
# 만약 로딩이 안된다면 root_dir을 직접 적어주세요.
# 데모 기준에서 OS별 root path는 아래와 같이 적으면 됩니다.
# windows : C:Users/yourname/yourdirectory/kochat/demo
# linux : /home/yourname/yourdirectory/kochat/demo

_ = '\\' if platform.system() == 'Windows' else '/'
if root_dir[len(root_dir) - 1] != _: root_dir += _

BASE = {
    'root_dir': root_dir.format(_=_),  # 백엔드 루트경로
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'vector_size': 128,  # 단어 벡터 사이즈
    'batch_size': 512,  # 미니배치 사이즈
    'max_len': 8,  # 문장의 최대 길이 (패드 시퀀싱)
    'delimeter': _,  # OS에 따른 폴더 delimeter

    'PAD': 0,  # PAD 토큰 값 (전체가 0인 벡터)
    'OOV': 1  # OOV 토큰 값 (전체가 1인 벡터)
}

DATA = {
    'data_ratio': 0.8,  # 학습\\검증 데이터 비율
    'raw_data_dir': BASE['root_dir'] + "data{_}raw{_}".format(_=_),  # 원본 데이터 파일 경로
    'ood_data_dir': BASE['root_dir'] + "data{_}ood{_}".format(_=_),  # out of distribution 데이터셋
    'intent_data_dir': BASE['root_dir'] + "data{_}intent_data.csv".format(_=_),  # 생성된 인텐트 데이터 파일 경로
    'entity_data_dir': BASE['root_dir'] + "data{_}entity_data.csv".format(_=_),  # 생성된 엔티티 데이터 파일 경로

    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'PLACE'],  # 사용자 정의 태그
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}

PROC = {
    'logging_precision': 5,  # 결과 저장시 반올림 소수점 n번째에서 반올림
    'model_dir': BASE['root_dir'] + "saved{_}".format(_=_),  # 모델 파일, 시각화 자료 저장 경로
    'visualization_epoch': 50,  # 시각화 빈도 (애폭마다 시각화 수행)
    'save_epoch': 100  # 저장 빈도 (에폭마다 모델 저장)
}

LOSS = {
    'center_factor': 0.025,  # Center Loss의 weighting 비율
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
    'sg': 1,  # 0 : CBOW = 1 \\ SkipGram = 2
    'iter': 3000  # 임베딩 학습 횟수
}

INTENT = {
    'model_lr': 1e-4,  # 인텐트 학습시 사용되는 러닝레이트
    'loss_lr': 1e-2,  # 인텐트 학습시 사용되는 러닝레이트
    'weight_decay': 1e-4,  # 인텐트 학습시 사용되는 가중치 감쇠 정도
    'epochs': 500,  # 인텐트 학습 횟수
    'd_model': 1024,  # 인텐트 모델의 차원
    'd_loss': 32,  # 인텐트 로스의 차원 (시각화차원, 높을수록 ood 디텍션이 정확해지지만 느려집니다.)
    'layers': 3,  # 인텐트 모델의 히든 레이어(층)의 수
    'grid_search': True,  # KNN과 Fallback Detector 학습시 그리드 서치 여부

    'lr_scheduler_factor': 0.75,  # 러닝레이트 스케줄러 감소율
    'lr_scheduler_patience': 10,  # 러닝레이트 스케줄러 감소 에폭
    'lr_scheduler_min_lr': 1e-12,  # 최소 러닝레이트
    'lr_scheduler_warm_up': 100,  # 러닝레이트 감소 시작시점

    # auto를 쓰려면 ood dataset을 함께 넣어줘야합니다.
    'distance_fallback_detection_criteria': 'auto',  # [auto, min, mean], auto는 OOD 데이터 있을때만 가능
    'distance_fallback_detection_threshold': -1,  # mean 혹은 min 선택시 임계값
    'softmax_fallback_detection_criteria': 'auto',  # [auto, other], auto는 OOD 데이터 있을때만 가능
    'softmax_fallback_detection_threshold': -1,  # other 선택시 fallback이 되지 않는 최소 값

    # 그리드 서치를 사용하지 않을때 KNN의 K값
    'num_neighbors': 10,

    # 그리드 서치를 사용할 때의 파라미터 목록
    'dist_param': {
        'n_neighbors': list(range(5, 15)),  # K값 범위 설정
        'weights': ["uniform"],  # ['uniform', 'distance']
        'p': [2],  # [1, 2] (맨하튼 vs 유클리디언)
        'algorithm': ['ball_tree']  # ['ball_tree', 'kd_tree']
    },

    # 폴백 디텍터 후보 (선형 모델을 추천합니다)
    'fallback_detectors': [
        LogisticRegression(max_iter=30000),
        LinearSVC(max_iter=30000)
    ]
}

ENTITY = {
    'model_lr': 1e-4,  # 엔티티 학습시 사용되는 모델 러닝레이트
    'loss_lr': 1e-4,  # 엔티티 학습시 사용되는 로스 러닝레이트 (아직 사용되지 않음)
    'weight_decay': 1e-4,  # 엔티티 학습시 사용되는 가중치 감쇠 정도
    'epochs': 1000,  # 엔티티 학습 횟수
    'd_model': 1024,  # 엔티티 모델의 차원
    'layers': 3,  # 엔티티 모델의 히든 레이어(층)의 수
    'masking': True,  # loss 계산시 패딩 마스크 여부

    'lr_scheduler_factor': 0.75,  # 러닝레이트 스케줄러 감소율
    'lr_scheduler_patience': 10,  # 러닝레이트 스케줄러 감소 에폭
    'lr_scheduler_min_lr': 1e-12,  # 최소 러닝레이트
    'lr_scheduler_warm_up': 100  # 러닝레이트 감소 시작시점
}

API = {
    'request_chat_url_pattern': 'request_chat',  # request_chat 기능 url pattern
    'fill_slot_url_pattern': 'fill_slot',  # fill_slot 기능 url pattern
    'get_intent_url_pattern': 'get_intent',  # get_intent 기능 url pattern
    'get_entity_url_pattern': 'get_entity'  # get_entity 기능 url pattern
}
