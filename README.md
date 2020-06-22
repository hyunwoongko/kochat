# Kochat

![kochat_main](docs/kochat.jpg)

Kochat은 한국어 전용 챗봇 프레임워크로, 자연어처리 개발자라면 
누구나 무료로 손쉽게 챗봇을 개발 할 수 있도록 돕는 오픈소스 소프트웨어입니다.
<br><br>

## Table of contents
- [1. 시작하기](#1.-시작하기)
- [2. 대화형 인터페이스 컨셉](#2.대화형-인터페이스-컨셉)
- [3. 컴포넌트](#1.-컴포넌트)
- [4. 사용법](#4.-사용법)

<br><br>

## 1.시작하기
준비중...

<br><br>

## 2. 대화형 인터페이스 컨셉
작성 예정.. (목적대화, 비목적대화, 인텐트, 엔티티 등 개념 설명하기)
<br><br>

## 3. 컴포넌트
작성 예정.. (UML 다이어그램 및 도식도 첨가하기)
<br><br>

## 4. 사용법

#### 4.1. Dataset
Dataset 클래스는 학습에 필요한 데이터셋을 생성하는 클래스입니다. 
여러개의 Raw File을 한개의 통합된 데이터 셋으로 만들고, 이를 전처리 및
임베딩하여 사용자 입장에서는 Vocabulary 관리나 텐서 사이즈 등을
신경 쓰지 않고 손쉽게 학습에 필요한 데이터 셋을 코드 몇 줄 만으로 세팅할 수 있습니다.
<br><br>

- 4.1.1. Dataset 생성하기
```python
prep = Preprocessor()
# 전처리기 객체 생성

dataset = Dataset(prep, ood=True)
# ood 데이터셋 사용시

dataset = Dataset(prep, ood=False)
# ood 데이터셋 미사용시
```
ood 데이터 셋은 Out of distribution의 약자로, 현재 개발하려는 도메인 이외의 의도가
담긴 데이터셋을 의미합니다. ood 데이터셋이 없어도 Kochat을 이용하는 데에는 전혀 지장이 없지만,
ood 데이터 셋이 갖춰지면 Fallback Detection의 Threshold를 자동으로 설정하고, 현재 구축한 모델의
Fallback Detection 성능을 validation할 수 있습니다. 또한 Dataset 클래스에는 데이터를
지속적으로 동기화하고, 특히 Entity 라벨에 존재하는 오타 및 실수를 자동으로 점검하여 사용자에게 보고합니다.

<br>

- 4.1.2. 학습을 위한 데이터셋 만들기
```python
embed_processor = GensimProcessor(
    model=EmbedFastText())
# 임베딩을 위한 임베딩 프로세서 생성

embed_dataset = dataset.load_embed()
# 임베딩 프로세서를 학습시키기 위한 데이터셋 빌드 및 로딩

embed_processor.fit(embed_dataset)
# 임베딩 프로세서 학습 (데이터 생성에 필수적입니다)

intent_dataset = dataset.load_intent(embed_processor)
# 인텐트 학습용 데이터셋 빌드 및 로딩

entity_dataset = dataset.load_entity(embed_processor)
# 엔티티 학습용 데이터셋 빌드 및 로딩
```
<br><br>

## 5. 실험 및 시각화
작성 예정.. (모델별 성능 그래프, 모델별 이슈, 2차원부터 N차원까지 각 로스함수별 피쳐스페이스 분포, CRF 성능비교 등 첨부하기)
<br><br>

## 6. 컨트리뷰터
작성 예정..
<br><br>

## 7. 기타 사항
#### 7.1. Kochat 제작 동기
이전에 여기저기서 코드를 긁어모아서 만든, 수준 낮은 제 딥러닝 chatbot 레포지토리가 
생각보다 큰 관심을 받으면서, 한국어로 된 챗봇 소스코드가 정말 많이 없다는 것을 느꼈습니다. 
현재 대부분의 한국어 챗봇 프레임워크들은 대부분 일반인을 겨냥하기 때문에 웹상에서
손쉬운 UX/UI 기반으로 서비스 중입니다. 일반인 사용자는 정말 사용하기 편리하겠지만,
저와 같은 자연어처리 개발자들은 모델도 커스터마이징 하고 싶고, 로스함수도 추가하고싶고, 
데이터 시각화도 하면서 더욱 높은 성능을 추구하고 싶지만 아쉽게도 한국어 챗봇 프레임워크 
중에는 이러한 방식으로 잘 알려진 프레임워크는 없습니다. <br><br>

그러던 중, 미국의 [RASA](https://rasa.com)라는 챗봇 프레임워크를 보게 되었습니다. RASA는 개발자가 직접 소스코드를
수정할 수 있기 때문에 다양한 부분을 커스터마이징 할 수 있습니다. 그러나 한국어를 제대로 지원하지 않아서, 
전용 토크나이저를 추가하는 등 매우 번거로운 작업이 필요하고 실제로 너무 다양한 컴포넌트가 존재하여 익숙해지는데 조금
어려운 편입니다. 때문에 저는 한국어 기반이면서 조금 더 컴팩트한 프레임워크를 제작하면 
챗봇을 빌드해야하는 자연어처리 개발자들에게 정말 유용할 것이라고 판단되어 이러한 프레임워크를 
제작하게 되었습니다. <br><br>

이름인 Kochat은 제 이름 앞 글자인 Ko와 한국어(Korean)의 앞글자인 Ko를 따와서 지었습니다.
Kochat은 앞으로도 계속 오픈소스 프로젝트로 유지될 것이며, 적어도 1달에 1번 이상은 새로운 모델을 추가하고, 
기존 소스코드의 버그를 수정하는 등 유지보수 작업을 이어갈 것이며 처음에는 미천한 실력인 제가 시작했지만,
그 끝은 RASA처럼 정말 유용하고 높은 성능을 보여주는 수준높은 오픈소스 프레임워크가 되었으면 좋겠습니다. :)
<br><br>

#### 7.2. TODO (앞으로 할 일)
- [ ] 간단한 웹 인터페이스 기반 데모 페이지 제작하기
- [x] 엔티티 학습에 CRF 및 로스 마스킹 추가하기 
- [ ] Transformer 기반 모델 추가 (한국어 BERT, GPT2)
- [ ] Jupyter Note Example 작성하기 + Binder 실행 환경
- [ ] 다양한 Seq2Seq 모델을 추가해서 Fallback시 대처할 수 있게 만들기
- [ ] 대화 흐름관리를 위한 Story기능 논문 조사하고 구현하기
- [ ] 크롤러 패키지의 개발을 자동화 할 방법 찾아보기 (중요)
<br><br>

## 7. 라이센스
```
Copyright 2020 Kochat.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```