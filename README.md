# Kochat
[![PyPI version](https://badge.fury.io/py/kochat.svg)](https://badge.fury.io/py/kochat)
![GitHub](https://img.shields.io/github/license/gusdnd852/kochat)
[![CodeFactor](https://www.codefactor.io/repository/github/gusdnd852/kochat/badge)](https://www.codefactor.io/repository/github/gusdnd852/kochat)
![01_introduction_kochat](https://user-images.githubusercontent.com/38183241/85958000-1b8ed080-b9cd-11ea-99d6-69b472f3e2ff.jpg)
<br><br><br>


## Table of contents

<br>

## 1. Kochat 이란?


**Kochat은 한국어 전용 챗봇 개발 프레임워크로, 머신러닝 개발자라면 
누구나 무료로 손쉽게 한국어 챗봇을 개발 할 수 있도록 돕는 오픈소스 프레임워크**입니다.
단순 Chit-chat이 아닌 사용자에게 여러 기능을 제공하는 상용화 레벨의 챗봇 개발은 
단일 모델만으로 개발되는 경우보다 다양한 데이터, configuration, ML모델, 
Restful Api 및 애플리케이션, 또 이들을 유기적으로 연결할 파이프라인을 갖추어야 하는데 
이 것을 처음부터 개발자가 스스로 구현하는 것은 굉장히 번거롭고 손이 많이 가는 작업입니다. 
실제로 챗봇 애플리케이션을 직접 구현하다보면 아래 그림처럼 실질적으로 모델 개발보다는 
이런 부분들에 훨씬 시간과 노력이 많이 필요합니다.
<br><br>

![01_introduction_mlcode](https://user-images.githubusercontent.com/38183241/85958001-1c276700-b9cd-11ea-8782-d521b4514cee.jpg)

Kochat은 이러한 부분을 해결하기 위해 제작되었습니다. 
데이터 전처리, 아키텍처, 모델과의 파이프라인, 실험 결과 시각화, 
성능평가 등은 Kochat의 구성을 사용하면서 개발자가 원하는 모델이나 Loss함수, 
데이터 셋 등만 간단하게 작성하여 내가 원하는 모델의 성능을 빠르게 실험할 수 있게 도와줍니다.
또한 프리 빌트인 모델들과 Loss 함수등을 지원하여 딥러닝이나 자연어처리에 대해 잘 모르더라도 
프로젝트에 데이터만 추가하면 손쉽게 상당히 높은 성능의 챗봇을 개발할 수 있게 도와줍니다. 
아직은 초기레벨이기 때문에 많은 모델과 기능을 지원하지는 않지만 점차 모델과 
기능을 늘려나갈 계획입니다.
<br><br>

### 1.1. 기존 챗봇 빌더와의 차이점
- 기존에 상용화된 많은 챗봇 빌더와 Kochat은 타깃으로 하는 사용자가 다릅니다.
상용화된 챗봇 빌더들은 매우 간편한 웹 기반의 UX/UI를 제공하며 일반인을 타깃으로 합니다.
그에 반해 **Kochat은 챗봇빌더 보다는 개발자를 타깃으로하는 프레임워크에 가깝습니다.**
개발자는 소스코드를 작성함에 따라서 프레임워크에 본인만의 모델을 추가할 수 있고, 
Loss 함수를 바꾸거나 본인이 원하면 아예 새로운 기능을 첨가할 수도 있습니다.

- **Kochat은 오픈소스 프로젝트입니다.** 따라서 많은 사람이 참여해서 함께 개발할 수 있고
만약 새로운 모델을 개발하거나 새로운 기능을 추가하고싶다면 얼마든지 레포지토리에 컨트리뷰션
할 수 있습니다.

- **Kochat은 무료입니다.** 매달 사용료를 내야하는 챗봇 빌더들에 비해 자체적인 서버만 
가지고 있다면 비용제약 없이 얼마든지 챗봇을 개발하고 서비스 할 수 있습니다. 
아직은 기능이 미약하지만 추후에는 정말 웬만한 챗봇 빌더들 보다 더 다양한 기능을 무료로 
제공할 예정입니다.
<br><br>

### 1.2. Kochat 제작 동기

![01_introduction_rasa](https://user-images.githubusercontent.com/38183241/85958002-1c276700-b9cd-11ea-8201-48976d8cf91d.png)

이전에 여기저기서 코드를 긁어모아서 만든, 수준 낮은 제 딥러닝 chatbot 레포지토리가 
생각보다 큰 관심을 받으면서, 한국어로 된 딥러닝 챗봇 구현체가 정말 많이 없다는 것을 느꼈습니다. 
현재 대부분의 챗봇 빌더들은 대부분 일반인을 겨냥하기 때문에 웹상에서 손쉬운 UX/UI 
기반으로 서비스 중입니다. 일반인 사용자는 사용하기 편리하겠지만, 저와 같은 개발자들은 
모델도 커스터마이징 하고 싶고, 로스함수도 바꿔보고싶고, 시각화도 하면서 더욱 높은 성능을 
추구하고 싶지만 아쉽게도 한국어 챗봇 빌더 중에서 이러한 방식으로 잘 알려진 것은 없습니다. 
<br><br>

그러던 중, 미국의 [RASA](https://rasa.com)라는 챗봇 프레임워크를 보게 되었습니다. 
RASA는 개발자가 직접 소스코드를 수정할 수 있기 때문에 다양한 부분을 커스터마이징 할 수 있습니다. 
그러나 한국어를 제대로 지원하지 않아서, 전용 토크나이저를 추가하는 등 매우 번거로운 작업이 
필요하고 실제로 너무 다양한 컴포넌트가 존재하여 익숙해지는데 조금 어려운 편입니다. 
때문에 누군가 한국어 기반이면서 조금 더 컴팩트한 챗봇 프레임워크를 제작한다면 
챗봇을 개발해야하는 개발자들에게 정말 유용할 것이라고 생각되었고 직접 이러한 프레임워크를 
만들어보자는 생각에 Kochat을 제작하게 되었습니다. <br><br>

Kochat은 한국어(Korean)의 앞글자인 Ko와 제 이름 앞 글자인 Ko를 따와서 지었습니다.
Kochat은 앞으로도 계속 오픈소스 프로젝트로 유지될 것이며, 적어도 1~2달에 1번 이상은 
새로운 모델을 추가하고, 기존 소스코드의 버그를 수정하는 등 유지보수 작업을 이어갈 것이며 
처음에는 미천한 실력인 제가 시작했지만, 그 끝은 RASA처럼 정말 유용하고 높은 성능을 보여주는 
수준높은 오픈소스 프레임워크가 되었으면 좋겠습니다. :)

<br><br><br>

## 2. About Chatbot 
이 챕터에서는 챗봇의 분류와 구현방법, Kochat은 어떻게 챗봇을 구현하고 있는지에 대해 
간단하게 소개합니다. 
<br><br>

### 2.1. 챗봇의 분류

![02_chatbot_table](https://user-images.githubusercontent.com/38183241/85957998-1af63a00-b9cd-11ea-8ed3-e3527fe790a7.jpg)

챗봇은 크게 비목적대화를 위한 Open domain 챗봇과 목적대화를 위한 Close domain 챗봇으로 나뉩니다.
Open domain 챗봇은 주로 잡담 등을 수행하는 챗봇을 의미하는데, 
여러분이 잘 알고있는 심심이 등이 챗봇이 대표적인 Open domain 챗봇이며 Chit-chat이라고도 불립니다.
Close domain 챗봇이란 한정된 대화 범위 안에서 사용자가 원하는 목적을 달성하기 위한 챗봇으로 
주로 금융상담봇, 식당예약봇 등이 이에 해당하며 Goal oriented 챗봇이라고도 불립니다. 
요즘 출시되는 시리나 빅스비 같은 인공지능 비서, 인공지능 스피커들은 특수 기능도 수행해야하고
사용자와 잡담도 잘 해야하므로 Open domain 챗봇과 Close domain 챗봇이 모두 포함되어 있는 경우가 많습니다.
<br><br>

### 2.2. 챗봇의 구현
챗봇을 구현하는 방법은 크게 통계기반의 챗봇과 딥러닝 기반의 챗봇으로 나뉩니다.
여기에서는 딥러닝 기반의 챗봇만 소개하도록 하겠습니다.
<br><br>
 
#### 2.2.1. Open domain 챗봇

![02_chatbot_seq2seq](https://user-images.githubusercontent.com/38183241/85957996-19c50d00-b9cd-11ea-8a86-8d814e737f45.png)

먼저 Open domain 챗봇의 경우는 딥러닝 분야에서는 대부분, End to End 
신경망 기계번역 방식(Seq2Seq)으로 구현되어왔습니다. Seq2Seq은 한 문장을 다른 문장으로 
변환/번역하는 방식입니다. 번역기에게 "나는 배고프다"라는 입력이 주어지면 "I'm Hungry"라고 
번역해내듯이, 챗봇 Seq2Seq는 "나는 배고프다"라는 입력이 주어질 때, "많이 배고프신가요?" 등의 대답으로 번역합니다. 
최근에 발표된 Google의 [Meena](https://ai.googleblog.com/2020/01/towards-conversational-agent-that-can.html)
같은 모델을 보면, 복잡한 모델 아키텍처나 학습 프레임워크 없이 End to End (Seq2Seq) 모델만으로도
매우 방대한 데이터셋과 높은 성능의 컴퓨팅 리소스를 활용하면 정말 사람과 근접한 수준으로 대화할 수 있다는 것으로 알려져있습니다.
(그러나 현재버전 프레임워크에서는 Close domain 만 지원합니다. 차후 버전에서 다양한 Seq2Seq 모델도 추가할 예정입니다.)
<br><br>

#### 2.2.2. Close domain 챗봇

![02_chatbot_slot_filling](https://user-images.githubusercontent.com/38183241/85957997-1a5da380-b9cd-11ea-9ead-9cb554efceaf.jpg)

Close domain 챗봇은 대부분 Slot Filling 방식으로 구현되어 왔습니다. 물론 Close domain 챗봇도
Open domain처럼 End to end로 구현하려는 [다양한](https://arxiv.org/pdf/1605.07683.pdf) 
[시도](https://arxiv.org/pdf/1702.03274.pdf) [들도](https://arxiv.org/pdf/1708.05956.pdf) 
[존재](https://arxiv.org/pdf/1804.08217.pdf) 하였으나, 논문에서 제시하는 
데이터셋에서만 잘 작동하고, 실제 다른 데이터 셋(Task6의 DSTC dataset)에 적용하면 그 정도의 
성능이 나오지 않았기 때문에 현업에 적용되기는 어려움이 있습니다. 때문에 현재는 대부분의 목적지향 
챗봇 애플리케이션이 기존 방식인 Slot Filling 방식으로 구현되고 있습니다.
<br><br>

Slot Filling 방식은 미리 기능을 수행할 정보를 담는 '슬롯'을 먼저 정의한 다음,
사용자의 말을 듣고 어떤 슬롯을 선택할지 정하고, 해당 슬롯을 채워나가는 방식입니다.
그리고 이러한 Slot Filling 방식 챗봇의 구현을 위해 '인텐트'와 '엔티티'라는 개념이 등장합니다.
말로만 설명하면 어려우니 예시를 봅시다. 가장 먼저 우리가 여행 정보 알림 챗봇을 만든다고 가정하고,
여행정보 제공을 위해 "날씨 정보제공", "미세먼지 정보제공", "맛집 정보제공", "여행지 정보제공"이라는 4가지 
핵심 기능을 구현해야한다고 합시다. 
<br><br>

#### 2.2.2.1. 인텐트(의도) 분류하기 : 슬롯 고르기

![02_chatbot_intent_classification](https://user-images.githubusercontent.com/38183241/85957993-1893e000-b9cd-11ea-858c-f0dd607f3825.jpg)

가장 먼저 사용자에게 문장을 입력받았을 때, 우리는 저 4가지 정보제공 기능 중
어떤 기능을 실행해야하는지 알아채야합니다. 이 것을 인텐트(Intent)분류. 즉, 의도 분류라고 합니다.
사용자로부터 "수요일 부산 날씨 어떠니?"라는 문장이 입력되면 4가지 기능 중  날씨 정보제공 기능을 
수행해야 한다는 것을 알아내야합니다. 때문에 문장 벡터가 입력되면, Text Classification을 수행하여
어떤 API를 사용해야할지 알아냅니다.
<br><br>

#### 2.2.2.2. 엔티티(개체명) 인식하기

![02_chatbot_entity_recognition](https://user-images.githubusercontent.com/38183241/85957992-17fb4980-b9cd-11ea-9a57-de36bc37a979.jpg)

그 다음 해야할 일은 바로 개체명인식 (Named Entity Recognition)입니다.
어떤 API를 호출할지 알아냈다면, 이제 그 API를 호출하기 위한 파라미터를 찾아야합니다.
만약 날씨 API의 실행을 위한 파라미터가 "지역"과 "날씨"라면 사용자의 입력 문장에서 "지역"에 관련된 정보와
"날씨"에 관련된 정보를 찾아내서 해당 슬롯을 채웁니다. 만약 사용자가 "수요일 날씨 알려줘"라고만 말했다면,
지역에 관련된 정보는 아직 찾아내지 못했기 때문에 다시 되물어서 찾아내야합니다. 
<br><br>

#### 2.2.2.3. 대답 생성하기

![02_chatbot_response_generation](https://user-images.githubusercontent.com/38183241/85957995-19c50d00-b9cd-11ea-8f88-50fea23df8d5.jpg)

슬롯이 모두 채워졌다면 API를 실행시켜서 외부로부터 정보를 제공받습니다.
API로부터 결과가 도착하면, 미리 만들어둔 템플릿 문장에 해당 실행 결과를 삽입하여 대답을 만들어내고,
이 대답을 사용자에게 response합니다. 이 API는 자유롭게 원하는 API를 사용하면 됩니다. 
예제 애플리케이션에서는 주로 웹 크롤링을 이용하여 API를 구성하였고, 크롤러 구현 아키텍처에 대해서도 후술하도록 하겠습니다. 
<br><br>

Slot Filling 방식의 챗봇은 위와 같은 흐름으로 진행됩니다. 따라서 이러한 방식의 챗봇을 구현하려면
최소한 3가지의 모듈이 필요합니다. 첫번째로 인텐트 분류모델, 엔티티 인식모델, 
그리고 대답 생성모듈(예제에서는 크롤링)입니다.
Kochat은 이 세가지 모듈과 이를 서빙할 Restful API까지 모두 포함하고 있습니다. 
이에 대해서는 "4. Usage" 챕터에서 각각 모델이 어떻게 구현되어 있는지 자세히 설명합니다.

<br><br><br>

## 3. Getting Started

### 3.1. Requirements 
Kochat을 이용하려면 반드시 본인의 OS와 머신에 맞는 Pytorch가 설치 되어있어야합니다.
만약 Pytorch를 설치하지 않으셨다면 [여기](https://pytorch.org/get-started/locally/) 에서 다운로드 받아주세요.
(Kochat을 설치한다고 해서 Pytorch가 함께 설치되지 않습니다. 본인 버전에 맞는 Pytorch를 다운로드 받아주세요)

<br>

### 3.2. pip install 
pip를 이용해 Kochat을 간단하게 다운로드하고 사용할 수 있습니다. 
아래 명령어를 통해서 kochat을 다운로드 받아주세요.
```shell script
pip install kochat
```

<br>

### 3.3 Dependencies
패키지를 구현하는데 사용된 디펜던시는 아래와 같습니다. 
(Kochat 설치시 함께 설치됩니다.)
```
matplotlib==3.2.1
pandas==1.0.4
gensim==3.8.3
konlpy==0.5.2
sklearn==0.0
numpy==1.18.5
joblib==0.15.1
scikit-learn==0.23.1
pytorch-crf==0.7.2
requests==2.24.0
flask==1.1.2
```

<br>

### 3.4 Configuration 파일 추가하기
pip를 이용해 Kochat을 내려받았다면 프로젝트에, kochat의 configuration 파일을 추가해야합니다.
[여기](https://github.com/gusdnd852/kochat/files/4843589/kochat_config.zip) 에서 Configuration파일을
다운로드 받고, 압축을 풀어서 interpreter의 working directory에 넣습니다.
(보통은 project root경로입니다. 만약 어떤 python 파일을 실행한다면, 그 실행파일과 동일한 경로에 반드시 
kochat_config.py가 있어야합니다.)

<br>

### 3.5 데이터셋 넣기
이제 여러분이 학습시킬 데이터셋을 넣어야합니다. 
그 전에 데이터셋의 포맷에 대해서 간단하게 알아봅시다. Kochat은 기본적으로 Slot filling을 기반으로
하고 있기 때문에 Intent와 Entity 데이터셋이 필요합니다. 그러나 이 두가지 데이터셋을 따로 만들면
상당히 번거로워지기 때문에 한가지 포맷으로 두가지 데이터를 자동으로 생성합니다.
아래와 같은 포맷으로 데이터셋을 만들어줍니다.

```
weather.csv

question,label
날씨 알려주세요,O O
월요일 인제 비오니,S-DATE S-LOCATION O
군산 날씨 추울까 정말,S-LOCATION O O O
곡성 비올까,S-LOCATION O
내일 단양 눈 오겠지 아마,S-DATE S-LOCATION O O O
강원도 춘천 가는데 오늘 날씨 알려줘,B-LOCATION E-LOCATION O S-DATE O O
전북 군산 가는데 화요일 날씨 알려줄래,B-LOCATION E-LOCATION O S-DATE O O
제주 서귀포 가려는데 화요일 날씨 알려줘,B-LOCATION E-LOCATION O S-DATE O O
오늘 제주도 날씨 알려줘,S-DATE S-LOCATION O O
... (생략)
```

```
travel.csv

question,label
어디 관광지 가겠냐,O O O
파주 유명한 공연장 알려줘,S-LOCATION O S-TRAVEL O
창원 여행 갈만한 바다,S-LOCATION O O S-TRAVEL
평택 갈만한 스키장 여행 해보고 싶네,S-LOCATION O S-TRAVEL O O O
제주도 템플스테이 여행 갈 데 추천해 줘,S-LOCATION S-TRAVEL O O O O O
전주 가까운 바다 관광지 보여줘 봐요,S-LOCATION O S-TRAVEL O O O
용인 가까운 축구장 어딨어,S-LOCATION O S-TRAVEL O
붐비는 관광지,O O
청주 가을 풍경 예쁜 산 가보고 싶어,S-LOCATION S-DATE O O S-TRAVEL O O
```

<br>

## 4. Usage

## 5. 실험 및 시각화

## 6. 컨트리뷰터

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
