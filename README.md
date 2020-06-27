# Kochat (Korean Chatbot)


[![CodeFactor](https://www.codefactor.io/repository/github/gusdnd852/kochat/badge/master)](https://www.codefactor.io/repository/github/gusdnd852/kochat/overview/master)
[![codebeat badge](https://codebeat.co/badges/e0c94e18-4127-4553-a576-a3cb28fdd925)](https://codebeat.co/projects/github-com-gusdnd852-kochat-master)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
![license](https://camo.githubusercontent.com/9de77196777ad799157befc0c599963dd909ce25/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f6166666a6c6a6f6f333538312f457870616e6461)
<br><br>

![kochat_main](docs/image1.jpg)

**Kochat은 한국어 전용 챗봇 개발 프레임워크로, 자연어처리 개발자라면 
누구나 무료로 손쉽게 챗봇을 개발 할 수 있도록 돕는 오픈소스 프레임워크**입니다.
단순 Chit-chat이 아닌 사용자에게 여러 기능을 제공하는 상용화 레벨의 챗봇 개발은 
단일 모델만으로 개발되는 경우보다 다양한 데이터, configuration, ML모델, 
Restful Api 및 애플리케이션, 또 이들을 유기적으로 연결할 파이프라인을 갖추어야 하는데 
이 것을 처음부터 개발자가 스스로 빌드하는 것은 굉장히 번거롭고 손이 많이 가는 작업입니다. 
때문에 챗봇 애플리케이션을 직접 구현하다보면 아래 그림처럼 실질적으로 모델 개발보다는 
이런 부분들에 훨씬 시간과 노력이 많이 필요합니다.
<br><br>

![kochat_main](docs/image2.jpg)

Kochat은 이러한 부분을 해결하기 위해 제작되었습니다. 
데이터 전처리, 아키텍처, 모델과의 파이프라인, 실험 결과 시각화, 성능평가 등은 
Kochat의 구성을 사용하면서 개발자가 원하는 모델이나 Loss함수, 데이터 셋 등만 
간단하게 작성하여 내가 원하는 모델의 성능을 빠르게 실험할 수 있게 도와줍니다.
또한 프리 빌트인 모델들과 Loss 함수등을 지원하여 데이터만 추가하면 
따로 모델을 만들지 않아도 손쉽게 상당히 높은 성능의 챗봇을 개발할 수 있게 도와줍니다. 
아직은 초기레벨이기 때문에 많은 모델과 기능을 지원하지는 않지만 점차 모델과 기능을 늘려나갈 계획입니다.
<br><br><br>

- ### 기존 챗봇 빌더와의 차이점
1. 기존에 상용화된 많은 챗봇 빌더과 Kochat은 타깃으로 하는 사용자가 다릅니다.
상용화된 챗봇 빌더들은 매우 간결한 UX/UI를 제공하며 일반인을 타깃으로 합니다.
그에 반해 **Kochat은 챗봇빌더 보다는 자연어처리 개발자를 타깃으로하는 프레임워크에 가깝습니다.**
개발자는 소스코드를 작성함에 따라서 프레임워크에 본인만의 모델을 추가할 수 있고, 
Loss 함수를 바꾸거나 본인이 원하면 아예 새로운 기능을 첨가할 수도 있습니다. 

2. **Kochat은 오픈소스 프로젝트입니다.** 따라서 많은 사람이 참여해서 함께 개발할 수 있고
만약 새로운 모델을 개발하거나 새로운 기능을 추가하고싶다면 얼마든지 레포지토리에 컨트리뷰션
할 수 있습니다. 기존 챗봇 빌더들보다 훨씬 자유도가 높고 개방적입니다.

3. **Kochat은 무료입니다.** 매달 사용료를 내는 챗봇 빌더들에 비해 자체적인 서버만 가지고 있다면
비용제약 없이 얼마든지 챗봇을 개발하고 서비스 할 수 있습니다.


<br><br>

## Table of contents
- [1. Kochat 시작하기](#1-kochat-시작하기)
    - [1.1 템플릿 레포지토리 만들기]()
- [2. 챗봇에 대한 간략한 설명](#2-챗봇에-대한-간략한-설명)
    - [2.1 챗봇의 분류](#21-챗봇의-분류)
- [3. 아키텍처와 컴포넌트](#3-아키텍처와-컴포넌트)
- [4. 라이브러리 사용법](#4-사용법)


<br><br>

## 1. Kochat 시작하기
우선 초기버전 코드는 Kochat 프레임워크 자체가 프레임워크로서 설계된 것이 
아니라 제가 구동해보고 정확하게 잘 돌아가는지에 초점을 맞춘 상태이기 때문에
템플릿 레포지토리로 제공하는 것이 저한테도 편리하고 프레임워크를 사용하는 
쪽에서도 편리할 것이라고 판단하였습니다. (configuraion, 파일 경로 등등 
처음부터 잡으려면 시간이 많이 들고 오류도 굉장히 잦을 것으로 판단되기 때문에..)
우선 지금은 템플릿 레포지토리로 제공하고, 추후에 기회가 되면 pip로도 
제공하도록 하겠습니다.
<br><br>

#### 1.1 템플릿 레포지토리 만들기

![kochat_main](docs/getting_started_01.jpg)

상단의 Use this template 버튼을 클릭하여 템플릿 레포지토리를 생성합니다.
<br><br>

![kochat_main](docs/getting_started_02.jpg)

레포지토리 이름과 설명 등의 정보를 기입한 뒤, 레포지토리를 생성합니다.
<br><br>

![kochat_main](docs/getting_started_03.jpg)

git clone 명령어를 사용해서 레포지토리를 clone합니다.
<br><br>

#### 1.2 configuration 설정하기

`_backed`패키지의 `config.py`에 데이터/모델 저장 경로 등 다양한 설정 값들이 있습니다.
레포지토리를 열고 나서 가장 먼저 이 설정 값을 변경합니다.

```python
"""
1. 여기에서 본인의 운영체제를 설정합니다. 

Windows의 경우 'Windows', 그 외의 경우 'Others' 
이는 파일 경로 delimeter 설정을 위해서 입니다. ('/' vs '\\')
"""

OS = 'Others' # or 'Windows'
_ = '\\' if OS == 'Windows' else '/'
```

OS를 설정했으면 다음 설정을 이어나갑니다.

```python
"""
2. 두번째로 레포지토리 root_dir을 설정합니다.
맨 뒤에 '_backend{_}.format(_=_)'를 꼭 붙이셔야 합니다!!

windows → "C:{_}yourdirectory{_}yourdirectory{_}..._backend{_}".format(_=_)
linux → "/home{_}yourdirectory{_}yourdirectory{_}..._backend{_}".format(_=_)
"""

BACKEND = {
    # ... 생략
    'root_dir': "/home{_}gusdnd852{_}Github{_}kochat{_}_backend{_}".format(_=_),  # 백엔드 루트경로
    # ... 생략
}
```

기본 configuration 설정은 모두 끝났습니다. 기본 설정 말고도 아래에 훨씬 많은 설정들이 있기 때문에
주석을 잘 보시고 원하는 부분의 설정값을 변경하셔서 사용하시길 바랍니다.
<br><br>


#### 1.3 데이터 삽입하기
이제 만들려는 챗봇의 데이터를 삽입합니다. 데이터는 `_backend/data/raw`폴더에 삽입합니다. 


## 2. 챗봇에 대한 간략한 설명
이 챕터에서는 챗봇의 분류와 구현방법, Kochat은 어떻게 챗봇을 구현하고 있는지에 대해 간단하게 소개합니다. 
<br><br>

#### 2.1 챗봇의 분류
챗봇은 크게 목적대화를 위한 Close domain 챗봇, 그리고 비목적대화를 위한 Open domain 챗봇으로 나뉩니다.
Close domain 챗봇이란 한정된 대화 범위 안에서 사용자가 원하는 목적을 달성하기 위한 챗봇으로 
주로 쇼핑몰 상담봇, 금융 상담봇, 

<br><br>

## 3. 아키텍처와 컴포넌트
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
