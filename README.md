# Kochat
[![PyPI version](https://badge.fury.io/py/kochat.svg)](https://badge.fury.io/py/kochat)
![GitHub](https://img.shields.io/github/license/gusdnd852/kochat)
[![CodeFactor](https://www.codefactor.io/repository/github/gusdnd852/kochat/badge)](https://www.codefactor.io/repository/github/gusdnd852/kochat)
![01_introduction_kochat](https://user-images.githubusercontent.com/38183241/85958000-1b8ed080-b9cd-11ea-99d6-69b472f3e2ff.jpg)
<br><br><br>


## Table of contents

<br><br><br>

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

#### 2.2.2.2. 폴백 검출하기 : 모르겠으면 모른다고 말하기
그러나 여기에 신경써야할 부분이 한 부분 존재합니다. 
일반적인 딥러닝 분류모델은 모델이 학습한 클래스 내에서만 분류가 가능합니다.
그러나 사용자가 네 가지의 발화의도 안에서만 말할 것이라는 보장은 없습니다.
만약 위처럼 "날씨 정보제공", "미세먼지 정보제공", "맛집 정보제공", "여행지 정보제공"의 데이터만
학습한 인텐트 분류모델에 "안녕 반갑다."라는 말을 하게 되면 어떻게 될까요? 위 4가지에 속하지 않은
발화 의도인 "인사"에 해당하지만 모델은 인삿말은 한번도 본적이 없기 때문에 이것도 역시 
4가지 중 하나로 분류하게 됩니다. 이러한 문제를 해결하기 위해 의도 분류모델에는 반드시
폴백 (Fallback) 검출 전략이 포함되어야합니다.

![fallback](https://user-images.githubusercontent.com/38183241/86210679-f7d7b000-bbaf-11ea-9083-d83cea2fe540.png)

보통의 챗봇빌더들은 입력 단어들의 임베딩인 문장 벡터와
기존 데이터셋에 있는 문장 벡터들의 Cosine 유사도를 비교합니다. 
이 때 가장 가까운 인접 클래스와의 각도가 임계치 이상이면 Fallback이고,
그렇지 않으면 가장 가까운 인접 클래스로 데이터 샘플을 분류하게 됩니다.
아래 그림을 보면 일반적인 챗봇 빌더들이 
어떤식으로 Fallback을 검출하는지 알 수 있습니다.

![fallback](https://user-images.githubusercontent.com/38183241/86210681-f908dd00-bbaf-11ea-8d8b-478f929a1f3a.png)

Kochat은 이렇게 단순히 문장들의 벡터 Cosine 유사도를 비교하지 않고 
더욱 고차원적인 방법을 사용하여 Fallback 디텍션을 보다 더 잘 수행하도록
설계하였는데 이에 대한 자세한 내용은 "4. Usage"에서 언급하도록 하겠습니다.
<br><br>

#### 2.2.2.3. 엔티티(개체명) 인식하기 : 슬롯 채우기

![02_chatbot_entity_recognition](https://user-images.githubusercontent.com/38183241/85957992-17fb4980-b9cd-11ea-9a57-de36bc37a979.jpg)

그 다음 해야할 일은 바로 개체명인식 (Named Entity Recognition)입니다.
어떤 API를 호출할지 알아냈다면, 이제 그 API를 호출하기 위한 파라미터를 찾아야합니다.
만약 날씨 API의 실행을 위한 파라미터가 "지역"과 "날씨"라면 사용자의 입력 문장에서 "지역"에 관련된 정보와
"날씨"에 관련된 정보를 찾아내서 해당 슬롯을 채웁니다. 만약 사용자가 "수요일 날씨 알려줘"라고만 말했다면,
지역에 관련된 정보는 아직 찾아내지 못했기 때문에 다시 되물어서 찾아내야합니다. 
<br><br>

#### 2.2.2.4. 대답 생성하기

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
numpy==1.18.5
joblib==0.15.1
scikit-learn==0.23.1
pytorch-crf==0.7.2
requests==2.24.0
flask==1.1.2
```

<br>

### 3.4 Configuration 파일 추가하기
pip를 이용해 Kochat을 다운로드 받았다면 프로젝트에, kochat의 configuration 파일을 추가해야합니다.
[kochat_config.zip](https://github.com/gusdnd852/kochat/files/4853859/kochat_config.zip) 을 
다운로드 받고 압축을 풀어서 interpreter의 working directory에 넣습니다. (kochat api를 실행하는 파일과
동일한 경로에 있어야합니다. 자세한 예시는 아래 데모에서 확인하실 수 있습니다.) 
config 파일에는 다양한 설정 값들이 존재하니 확인하고 입맛대로 변경하시면 됩니다.

<br>

### 3.5 데이터셋 넣기
이제 여러분이 학습시킬 데이터셋을 넣어야합니다. 
그 전에 데이터셋의 포맷에 대해서 간단하게 알아봅시다. 
Kochat은 기본적으로 Slot filling을 기반으로
하고 있기 때문에 Intent와 Entity 데이터셋이 필요합니다. 
그러나 이 두가지 데이터셋을 따로 만들면 상당히 번거로워지기 때문에 
한가지 포맷으로 두가지 데이터를 자동으로 생성합니다.
아래 데이터셋 규칙들에 맞춰서 데이터를 생성해주세요
<br><br>

#### 3.5.1. 데이터 포맷
기본적으로 intent와 entity를 나누려면, 두가지를 모두 구분할 수 있어야합니다.
그래서 선택한 방식은 인텐트는 파일로 구분, 엔티티는 라벨로 구분하는 것이였습니다.
추후 릴리즈 버전에서는 Rasa처럼 훨씬 쉬운 방식으로 변경하려고 합니다만, 초기버전에서는
다소 불편하더라도 아래의 포맷을 따라주시길 바랍니다. <br>

- weather.csv
```
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
- travel.csv
```
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
... (생략)
```

위 처럼 question,label이라는 헤더(컬럼명)을 가장 윗줄에 위치시키고,
그 아래로 두개의 컬림 question과 label에 해당하는 내용을 작성합니다.
각 단어 및 엔티티는 띄어쓰기로 구분됩니다.
예시 데이터는 BIO태깅을 개선한 BIOES태깅을 사용하여 라벨링했는데, 엔티티 태깅 방식은 자유롭게
고르셔도 됩니다. (config에서 설정 가능합니다.) 엔티티 태깅 스키마에 관련된 자세한 내용은 
[여기](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) 를 참고하세요.

<br>

#### 3.5.2. 데이터셋 저장 경로
데이터셋 저장경로는 기본적으로 config파일이 있는 곳을 root로 생각했을 때,
"root/data/raw"입니다. 이 경로는 config의 DATA 챕터에서 변경 가능합니다.
```
root
  |_data
    |_raw
      |_weather.csv
      |_dust.csv
      |_retaurant.csv
      |_...
```
<br>

#### 3.5.3. 인텐트 단위로 파일 분할
각 인텐트 단위로 파일을 분할합니다. 이 때, 파일명이 인텐트명이 됩니다.
파일명은 한글로 해도 상관 없긴 하지만, 리눅스 운영체제의 경우 시각화시 
matplotlib에 한글폰트가 설치되어있지 않다면 글자가 깨지니,
가급적이면 시각화를 위해 영어로 하는 것을 권장합니다. 
(만약 글자가 깨지지 않으면 한글로 해도 무방하니, 한글로 하려면 폰트를 설치해주세요.)
```
root
  |_data
    |_raw
      |_weather.csv      ← intent : weather
      |_dust.csv         ← intent : dust
      |_retaurant.csv    ← intent : restaurant
      |_...
```
<br>

#### 3.5.4. 파일의 헤더(컬럼명) 설정
파일의 헤더(컬럼명)은 반드시 question과 label로 해주세요.
헤더를 config에서 바꿀 수 있게 할까도 생각했지만, 
별로 큰 의미가 없는 것 같아서
우선은 고정된 값인 question과 label로 설정하였습니다.
```
question,label ← 중요 !!!
... (생략)
```
<br>

#### 3.5.4. 라벨링 실수 검출기능
샘플 당 question의 단어 갯수와 label의 엔티티 갯수는 동일해야하며 config에 정의한 엔티티만 사용 가능합니다.
이러한 라벨링 실수는 Kochat이 데이터를 변환할때 검출해서 어디가 틀렸는지 알려줍니다.

```
case 1: 라벨링 매칭 실수 방지


question = 전주 눈 올까 (size : 3)
label = S-LOCATION O O O (size : 4)

→ 에러 발생! (question과 label의 수가 다름)
```

```
case 2: 라벨링 오타 방지


(in kochat_config.py)
DATA = {
    ... (생략)

    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'TRAVEL'],  # 사용자 정의 태그
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}

question = 전주 눈 올까
label = Z-LOC O O

→ 에러 발생! (정의되지 않은 엔티티 : Z-LOC)
NER_tagging + '-' + NER_categories의 형태가 아니면 에러를 반환합니다.
```
<br>

#### 3.5.5. OOD 데이터셋
OOD란 Out of distribution의 약자로, 분포 외 데이터셋을 의미합니다.
즉, 현재 챗봇이 지원하는 기능 이외의 데이터를 의미합니다.
OOD 데이터셋이 없어도 Kochat을 이용하는데에는 아무런 문제가 없지만,
OOD 데이터셋을 갖추면 매우 귀찮은 몇몇 부분들을 효과적으로 자동화 할 수 있습니다. 
(주로 Fallback Detection threshold 설정)
OOD 데이터셋은 아래처럼 "root/data/ood"에 추가합니다.

```
root
  |_data
    |_raw
      |_weather.csv      
      |_dust.csv         
      |_retaurant.csv
      |_...
    |_ood
      |_ood_data_1.csv    ← data/ood폴더에 위치하게 합니다.
      |_ood_data_2.csv    ← data/ood폴더에 위치하게 합니다.
```
<br>

OOD 데이터셋은 아래와 같이 question과 OOD의 의도로 라벨링합니다.
예시 데이터셋은 전부 의도대로 라벨링했지만, 이 의도값을 사용하진 않기 때문에
그냥 아무값으로나 라벨링해도 사실 무관합니다.

```
예시_ood_데이터.csv

question,label
최근 있던일 최근 이슈 알려줘,뉴스이슈
최근 핫했던 것 알려줘,뉴스이슈
나한테 좋은 명언해줄 수 있냐,명언
나 좋은 명언 좀 들려주라,명언
좋은 명언 좀 해봐,명언
백재범 노래 들을래요,음악
비 노래 깡 듣고 싶다,음악
영화 ost 추천해줘,음악
지금 시간 좀 알려달라고,날짜시간
지금 시간 좀 알려줘,날짜시간
지금 몇 시 몇 분인지 아니,날짜시간
명절 스트레스 ㅜㅜ,잡담
뭐하고 놀지 ㅎㅎ,잡담
나랑 놀아주라 좀,잡담
뭐하고 살지,잡담
... (생략)
```
<br>

이렇게 라벨링 해도 되지만 어차피 라벨 데이터를 사용하지 않기 때문에 아래처럼 라벨링해도 무관합니다.
```
예시_ood_데이터.csv

question,label
최근 있던일 최근 이슈 알려줘,OOD
최근 핫했던 것 알려줘,OOD
나한테 좋은 명언해줄 수 있냐,OOD
나 좋은 명언 좀 들려주라,OOD
좋은 명언 좀 해봐,OOD
백재범 노래 들을래요,OOD
비 노래 깡 듣고 싶다,OOD
영화 ost 추천해줘,OOD
지금 시간 좀 알려달라고,OOD
지금 시간 좀 알려줘,OOD
지금 몇 시 몇 분인지 아니,OOD
명절 스트레스 ㅜㅜ,OOD
뭐하고 놀지 ㅎㅎ,OOD
나랑 놀아주라 좀,OOD
뭐하고 살지,OOD
... (생략)
```

OOD 데이터는 물론 많으면 좋겠지만 만드는 것 자체가 부담이기 때문에 적은 수만 넣어도 됩니다.
예시 데이터의 경우는 총 3000라인의 데이터 중 600라인정도의 OOD 데이터를 삽입하였습니다.
추후 버전에서는 가벼운 N-gram 기법(마르코프 체인 등)을 이용하여 OOD 데이터 생성을 자동화
할 계획입니다. 데이터까지 모두 삽입하셨다면 kochat을 이용할 준비가 끝났습니다. 아래 챕터에서는 
자세한 사용법에 대해 알려드리겠습니다.
<br><br><br>

## 4. Usage
### 4.1. `from kochat.data`
`kochat.data` 패키지에는 `Dataset` 클래스가 있습니다. `Dataset`클래스는 
분리된 raw 데이터 파일들을 하나로 합쳐서 통합 intent파일과 통합 entity파일로 만들고, 
embedding, intent, entity, inference에 관련된 데이터셋을 미니배치로 잘라서 
pytorch의 `DataLoader`형태로 제공합니다. 
또한 모델, Loss 함수 등을 생성할 때 파라미터로 입력하는 `label_dict`를 제공합니다.
`Dataset` 클래스를 생성할 때 필요한 파라미터인 `ood`는 OOD 데이터셋 사용 여부입니다. 
True로 설정하면 ood 데이터셋을 사용합니다. 

<br>

- Dataset 기능 1. 데이터셋 생성
```python
from kochat.data import Dataset


# 클래스 생성시 raw파일들을 검증하고 통합합니다.
dataset = Dataset(ood=True, naver_fix=True)  

# 임베딩 데이터셋 생성
embed_dataset = dataset.load_embed() 

# 인텐트 데이터셋 생성 (임베딩 프로세서 필요)
intent_dataset = dataset.load_intent(emb) 

# 엔티티 데이터셋 생성 (임베딩 프로세서 필요)
entity_dataset = dataset.load_entity(emb) 

# 추론용 데이터셋 생성 (임베딩 프로세서 필요)
predict_dataset = dataset.load_predict("서울 맛집 추천해줘", emb) 
```
<br>

- Dataset 기능 2. 라벨 딕셔너리 생성
```python
from kochat.data import Dataset


# 클래스 생성시 raw파일들을 검증하고 통합합니다.
dataset = Dataset(ood=True, naver_fix=True)  

# 인텐트 라벨 딕셔너리를 생성합니다.
intent_dict = dataset.intent_dict 

# 엔티티 라벨 딕셔너리를 생성합니다.
entity_dict = dataset.entity_dict
```
<br>

#### ⚠ Warning

`Dataset`클래스는 전처리시 토큰화를 수행할 때,
학습/테스트 데이터는 띄어쓰기를 기준으로 토큰화를 수행하고, 실제 사용자의 입력에
추론할 때는 네이버 맞춤법 검사기와 Konlpy 토크나이저를 사용하여 토큰화를 수행합니다.
네이버 맞춤법 검사기를 사용하면 성능은 더욱 향상되겠지만, 상업적으로 이용시 문제가
발생할 수 있고, 이에 대해 개발자는 어떠한 책임도 지지 않습니다.  <br><br>


만약 Kochat을 상업적으로 이용하시려면 `Dataset` 생성시 `naver_fix`파라미터를 
`False`로 설정해주시길 바랍니다. `False` 설정시에는 Konlpy 토큰화만 수행하며,
추후 버전에서는 네이버 맞춤법 검사기를 자체적인 띄어쓰기 검사모듈 등으로 
교체할 예정입니다.
<br><br><br>

### 4.2. `from kochat.model`
`model` 패키지는 사전 정의된 다양한 built-in 모델들이 저장된 패키지입니다.
현재 버전에서는 아래 목록에 해당하는 모델들을 지원합니다. 추후 버전이 업데이트 되면
지금보다 훨씬 다양한 built-in 모델을 지원할 예정입니다. 아래 목록을 참고하여 사용해주시길 바랍니다.

<br>

#### 4.2.1. embed 모델
```python
from kochat.model import embed


# 1. Gensim의 Word2Vec 모델의 Wrapper입니다.
# (OOV 토큰의 값은 config에서 설정 가능합니다.)
word2vec = embed.Word2Vec()

# 2. Gensim의 FastText 모델의 Wrapper입니다.
fasttext = embed.FastText()
```
<br>

#### 4.2.2. intent 모델
```python
from kochat.model import intent


# 1. Residual Learning을 지원하는 1D CNN입니다.
cnn = intent.CNN(label_dict=dataset.intent_dict, residual=True)

# 2. Bidirectional을 지원하는 LSTM입니다.
lstm = intent.LSTM(label_dict=dataset.intent_dict, bidirectional=True)
```
<br>

#### 4.2.3. entity 모델
```python
from kochat.model import entity


# 1. Bidirectional을 지원하는 LSTM입니다.
lstm = entity.LSTM(label_dict=dataset.entity_dict, bidirectional=True)
```
<br>

#### 4.2.4. 커스텀 모델
Kochat은 커스텀 모델을 지원합니다. 
Gensim이나 Pytorch로 작성한 커스텀 모델을 직접 학습시키기고 챗봇 애플리케이션에 
사용할 수 있습니다. 그러나 만약 커스텀 모델을 사용하려면 아래의 몇가지 규칙을 반드시 
따라야합니다.
<br><br>

#### 4.2.4.1. 커스텀 Gensim embed 모델
임베딩의 경우 현재는 Gensim 모델만 지원합니다. 추후에 Pytorch로 된
임베딩 모델(ELMO, BERT)등도 지원할 계획입니다.
Gensim Embedding 모델은 아래와 같은 형태로 구현해야합니다.
<br><br>

1. `@gensim` 데코레이터 설정
2. `BaseWordEmbeddingsModel`모델 중 한 가지 상속받기
4. `super().__init__()`에 파라미터 삽입하기 (self.XXX로 접근가능)
<br><br>

```python
from gensim.models import FastText
from kochat.decorators import gensim

# 1. @gensim 데코레이터를 설정하면 
# config의 GENSIM에 있는 모든 데이터에 접근 가능합니다.

@gensim
class FastText(FastText):
# 2. BaseWordEmbeddingsModel 모델중 한 가지를  상속받습니다.

    def __init__(self):
        # 3. `super().__init__()`에 필요한 파라미터를 넣어서 초기화해줍니다.

        super().__init__(size=self.vector_size,
                         window=self.window_size,
                         workers=self.workers,
                         min_count=self.min_count,
                         iter=self.iter)
```
<br><br>

#### 4.2.4.2. 커스텀 Intent 모델
인텐트 모델은 torch로 구현합니다.
인텐트 모델에는 `self.label_dict` 가 반드시 존재해야합니다. 
또한 최종 output 레이어는 자동생성되기 때문에 feature만 출력하면 됩니다.
더욱 세부적인 규칙은 다음과 같습니다.
<br><br>

1. `@intent` 데코레이터 설정
2. `torch.nn.Module` 상속받기
3. 파라미터로 label_dict를 입력받고 `self.label_dict`에 할당하기
4. `forward()` 함수에서 feature를 [batch_size, -1] 로 만들고 리턴
<br><br>

```python
from torch import nn
from torch import Tensor
from kochat.decorators import intent
from kochat.model.layers.convolution import Convolution


# 1. @intent 데코레이터를 설정하면 
# config의 INTENT에 있는 모든 설정값에 접근 가능합니다.

@intent
class CNN(nn.Module):
# 2. torch.nn의 Module을 상속받습니다.

    def __init__(self, label_dict: dict, residual: bool = True):
        super(CNN, self).__init__()
        self.label_dict = label_dict
        # 3. intent모델은 반드시 속성으로 self.label_dict를 가지고 있어야합니다.

        self.stem = Convolution(self.vector_size, self.d_model, kernel_size=1, residual=residual)
        self.hidden_layers = nn.Sequential(*[
            Convolution(self.d_model, self.d_model, kernel_size=1, residual=residual)
            for _ in range(self.layers)])

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.hidden_layers(x)

        return x.view(x.size(0), -1)
        # 4. feature를 [batch_size, -1]로 만들고 반환합니다.
        # 최종 output 레이어는 kochat이 자동 생성하기 때문에 feature만 출력합니다.
````
```python
import torch
from torch import nn, autograd
from torch import Tensor
from kochat.decorators import intent


# 1. @intent 데코레이터를 설정하면 
# config의 INTENT에 있는 모든 설정값에 접근 가능합니다.

@intent
class LSTM(nn.Module):
# 2. torch.nn의 Module을 상속받습니다.
 
    def __init__(self, label_dict: dict, bidirectional: bool = True):

        super().__init__()
        self.label_dict = label_dict
        # 3. intent모델은 반드시 속성으로 self.label_dict를 가지고 있어야합니다.

        self.direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=self.vector_size,
                            hidden_size=self.d_model,
                            num_layers=self.layers,
                            batch_first=True,
                            bidirectional=bidirectional)

    def init_hidden(self, batch_size: int) -> autograd.Variable:
        param1 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        param2 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        return autograd.Variable(param1), autograd.Variable(param2)

    def forward(self, x: Tensor) -> Tensor:
        b, l, v = x.size()
        out, (h_s, c_s) = self.lstm(x, self.init_hidden(b))

        # 4. feature를 [batch_size, -1]로 만들고 반환합니다.
        # 최종 output 레이어는 kochat이 자동 생성하기 때문에 feature만 출력합니다.
        return h_s[0]
```
<br><br>

#### 4.2.4.3. 커스텀 Entity 모델
엔티티 모델도 역시 torch로 구현합니다.
엔티티 모델에도 역시 `self.label_dict` 가 반드시 존재해야하며, 
또한 최종 output 레이어는 자동생성되기 때문에 feature만 출력하면 됩니다.
더욱 세부적인 규칙은 다음과 같습니다.
<br><br>

1. `@entity` 데코레이터 설정
2. `torch.nn.Module` 상속받기
3. 파라미터로 label_dict를 입력받고 `self.label_dict`에 할당하기
4. `forward()` 함수에서 feature를 [batch_size, max_len, -1] 로 만들고 리턴
<br><br>

```python
import torch
from torch import nn, autograd
from torch import Tensor
from kochat.decorators import entity

# 1. @entity 데코레이터를 설정하면 
# config의 ENTITY에 있는 모든 설정값에 접근 가능합니다.

@entity
class LSTM(nn.Module):
# 2. torch.nn의 Module을 상속받습니다.
 
    def __init__(self, label_dict: dict, bidirectional: bool = True):

        super().__init__()
        self.label_dict = label_dict
        # 3. entity모델은 반드시 속성으로 self.label_dict를 가지고 있어야합니다.

        self.direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=self.vector_size,
                            hidden_size=self.d_model,
                            num_layers=self.layers,
                            batch_first=True,
                            bidirectional=bidirectional)

    def init_hidden(self, batch_size: int) -> autograd.Variable:
        param1 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        param2 = torch.randn(self.layers * self.direction, batch_size, self.d_model).to(self.device)
        return torch.autograd.Variable(param1), torch.autograd.Variable(param2)

    def forward(self, x: Tensor) -> Tensor:
        b, l, v = x.size()
        out, _ = self.lstm(x, self.init_hidden(b))

        # 4. feature를 [batch_size, max_len, -1]로 만들고 반환합니다.
        # 최종 output 레이어는 kochat이 자동 생성하기 때문에 feature만 출력합니다.
        return out
```
<br><br><br>

### 4.3. `from kochat.proc`
`proc`은 Procssor의 줄임말로, 다양한 모델들의 
학습/테스트을 수행하는 함수인 `fit()`과
추론을 수행하는 함수인 `predict()` 등을 수행하는 클래스 집합입니다.
현재 지원하는 프로세서는 총 4가지로 아래에서 자세하게 설명합니다.
<br><br>

#### 4.3.1. `from kochat.proc import GensimEmbedder`
GensimEmbedder는 Gensim의 임베딩 모델을 학습시키고,
학습된 모델을 사용해 문장을 임베딩하는 클래스입니다. 자세한 사용법은 다음과 같습니다.

```python
from kochat.data import Dataset
from kochat.proc import GensimEmbedder
from kochat.model import embed


dataset = Dataset(ood=True)

# 프로세서 생성
emb = GensimEmbedder(
    model=embed.FastText()
)

# 모델 학습
emb.fit(dataset.load_embed())

# 모델 추론 (임베딩)
user_input = emb.predict("서울 홍대 맛집 알려줘")
```
<br><br>

#### 4.3.2. `from kochat.proc import SoftmaxClassifier`
`SoftmaxClassifier`는 가장 기본적인 Intent 분류 프로세서입니다.
이름이 SoftmaxClassifier인 이유는 Softmax Score를 이용해 Fallback Detection을 수행하기 때문에
이렇게 명명하게 되었습니다. 그러나 몇몇 [논문](https://arxiv.org/abs/1412.1897)
에서 Calibrate되지 않은 Softmax Score을 마치 Confidence처럼
착각해서 사용하면 심각한 문제가 발생할 수 있다는 것을 보여주었습니다. 

![mnist](https://user-images.githubusercontent.com/38183241/86215372-784ddf00-bbb7-11ea-8370-f1ab148e92e4.png)

<br>

위의 그림은 MNIST 분류모델에서 0.999 이상의 Softmax Score를 가지는 이미지들입니다.
실제로 0 ~ 9까지의 숫자와는 전혀 상관없는 이미지들이기 때문에 낮은 Softmax Score를
가질 것이라고 생각되지만 실제로는 그렇지 않습니다. 
사실 `SoftmaxClassifier`를 실제 챗봇의 Intent Classification 기능을 위해
사용하는 것은 적절하지 못합니다. `SoftmaxClassifier`는 아래 후술할 `DistanceClassifier`
와의 성능 비교를 위해, 또 아래에 후술할 Kochat NLU 기능 중 한 가지로 제공하기 위해 구현하였습니다.
사용법은 아래와 같습니다.

```python
from kochat.data import Dataset
from kochat.proc import SoftmaxClassifier
from kochat.model import intent
from kochat.loss import CrossEntropyLoss


dataset = Dataset(ood=True)

# 프로세서 생성
clf = SoftmaxClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CrossEntropyLoss(dataset.intent_dict)
)

# 되도록이면 SoftmaxClassifier는 CrossEntropyLoss를 이용해주세요
# 다른 Loss 함수들은 거리 기반의 Metric Learning을 수행하기 때문에 
# Softmax Classifiaction에 적절하지 못할 수 있습니다.


# 모델 학습
clf.fit(dataset.load_intent(emb))

# 모델 추론 (임베딩)
clf.predict(dataset.load_predict("오늘 서울 날씨 어떨까", emb))
```

<br>

#### 4.3.2. `from kochat.proc import DistanceClassifier`
`DistanceClassifier`는 `SoftmaxClassifier`와는 다르게 거리기반으로 작동하며,
일종의 Memory Network입니다. [batch_size, -1] 의 사이즈로 출력된 출력벡터와 기존 데이터셋에 있는 벡터들 사이의
거리를 계산하여 데이터셋에서 가장 가까운 K개의 샘플을 찾고 최다 샘플 클래스로 분류하는 
최근접 이웃 Retrieval 기반의 분류 모델입니다.
<br><br>

이 때 각 클래스들 사이의 거리가 멀어야 분류하는데 좋기 때문에 사용자가 설정한 
Loss함수(주로 Margin 기반 Loss)를 적용해 Metric Learning을 수행해서 클래스 간의 
Margin을 최대치로 벌리는 메커니즘이 구현되어있습니다.
또한 최근접 이웃 알고리즘의 K값은 config에서 직접 지정 할 수도 있고 
GridSearch를 적용하여 자동으로 최적의 K값을 찾을 수 있게 설계하였습니다. 
최근접 샘플을 찾을 때 Brute force하게 직접 거리를 일일이 다 구하면 굉장히 느리기 
때문에 다차원 검색트리인 `KDTree` 혹은 `BallTree` (KDTree의 개선 형태)를 통해서 
거리를 계산하며 결과로 만들어진 트리 구조를 메모리에 저장합니다. 
트리기반의 검색 알고리즘을 사용하기 때문에 `SoftmaxClassifier`와 
거의 비슷한 속도로 학습 및 추론이 가능합니다. 사용법은 아래와 같습니다.


```python
from kochat.data import Dataset
from kochat.proc import DistanceClassifier
from kochat.model import intent
from kochat.loss import CenterLoss


dataset = Dataset(ood=True)

# 프로세서 생성
clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict)
)

# 되도록이면 DistanceClassifier는 Margin 기반의 Loss 함수를 이용해주세요
# 현재는 CenterLoss, COCOLoss, Cosface, GausianMixture 등의 Metric Learning용 
# Loss함수를 지원합니다. Loss 함수별 성능 비교 및 피쳐분포는 아래에 첨부할 예정입니다.


# 모델 학습
clf.fit(dataset.load_intent(emb))

# 모델 추론 (임베딩)
clf.predict(dataset.load_predict("오늘 서울 날씨 어떨까", emb))
```
<br><br>

#### 4.3.3. `FallbackDetector (Classifier에 내장됨)`
`SoftmaxClassifier`와 `DistanceClassifier` 모두 Fallback Detection 기능을 구현되어있습니다.
Fallback Detection 기능을 이용하는 방법은 아래와 같이 두 가지 방법을 제공합니다.

```
1. OOD 데이터가 없는 경우 : 직접 config의 Threshold를 맞춰야합니다.
2. OOD 데이터가 있는 경우 : 머신러닝을 이용하여 Threshold를 자동 학습합니다.
```

<br>

바로 여기에서 OOD 데이터셋이 사용됩니다. 
`SoftmaxClassifier`는 out distribution 샘플들과 in distribution 샘플간의 
maximum softmax score (size = [batch_size, 1])를 feature로 하여 
머신러닝 모델을 학습하고, 
`DistanceClassifier`는 out distribution 샘플들과 in distribution 샘플들의 
K개의 최근접 이웃의 거리 (size = [batch_size, K])를 feature로 하여 
머신러닝 모델을 학습합니다. 

<br>

이러한 머신러닝 모델을 `FallbackDetector`라고 합니다. `FallbackDetector`는 각 
Classifier안에 내장 되어있기 때문에 별다른 추가 소스코드 없이 `Dataset`의 `ood` 
파라미터만 `True`로 설정되어있다면 Classifier학습이 끝나고나서 자동으로 학습되고, 
`predict()`시 저장된 `FallbackDetector`가 있다면 자동으로 동작합니다.
또한 `FallbackDetector`로 사용할 모델은 아래처럼 config에서 사용자가 직접 설정할 수 있으며
GridSearch를 지원하여 여러개의 모델을 리스트에 넣어두면 Kochat 프레임워크가
현재 데이터셋에 가장 적합한 `FallbackDetector`를 자동으로 골라줍니다. 

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


INTENT = {
    # ... (생략)

    # 폴백 디텍터 후보 (선형 모델을 추천합니다)
    'fallback_detectors': [
        LogisticRegression(max_iter=10000),
        LinearSVC(max_iter=10000)

        # 가능한 max_iter를 높게 설정해주세요
    ]
}
```
<br>

Fallback Detection 문제는 Fallback 메트릭(거리 or score)가 일정 임계치를 넘어가면 
샘플을 in / out distribution 샘플로 분류하는데 그 임계치를 현재 모르는 상황이므로 
선형 문제로 해석할 수 있습니다. 따라서 FallbackDetector로는 위 처럼 선형 모델인 
선형 SVM, 로지스틱 회귀 등을 주로 이용합니다. 물론 위의 리스트에 
`RandomForestClassifier()`나 `BernoulliNB()`, `GradientBoostingClassifier()` 등
다양한 sklearn 모델을 입력해도 동작은 하지만, 일반적으로 선형모델이 가장 우수하고 
안정적인 성능을 보였습니다. (이에 대한 실험결과는 아래에 첨부할 예정입니다.)

<br>

이렇게 Fallback의 메트릭으로 머신러닝 모델을 학습하면 Threshold를 직접 유저가 
설정하지 않아도 됩니다. OOD 데이터셋이 필요하다는 치명적인 단점이 있지만, 
차후 버전에서는 BERT와 Markov Chain을 이용해 OOD 데이터셋을 자동으로 빠르게 생성하는 
모델을 구현하여 추가할 예정입니다. (업데이트 이후부터는 OOD 데이터셋이 필요 없어집니다.)

<br>

그러나 아직 OOD 데이터셋 생성기능은 지원하지 않기 때문에 
현재 버전에서는 만약 OOD 데이터셋이 없다면 사용자가 직접 Threshold를 설정해야 하므로 눈으로
샘플들이 어느정도 score 혹은 거리를 갖는지 확인해야합니다. 따라서 Kochat은 Calibrate 모드를 지원합니다.

```python
while True:
    user_input = dataset.load_predict(input(), emb) 
    # 터미널에 직접 ood로 생각될만한 샘플을 입력해서 
    # 눈으로 결과를 직접 확인하고, threshold를 직접 조정합니다.

    result = clf.predict(user_input, calibrate=True)
    print("classification result : {}".format(result))


# DistanceClassifier
>>> '=====================CALIBRATION_MODE====================='
    '현재 입력하신 문장과 기존 문장들 사이의 거리 평균은 2.912이고'
    '가까운 샘플들과의 거리는 [2.341, 2.351, 2.412, 2.445 ...]입니다.'
    '이 수치를 보고 Config의 fallback_detection_threshold를 맞추세요.'
    'criteria는 거리평균(mean) / 최솟값(min)으로 설정할 수 있습니다.'


# SoftmaxClassifier
>>> '=====================CALIBRATION_MODE====================='
    '현재 입력하신 문장의 softmax logits은 0.997입니다.'
    '이 수치를 보고 Config의 fallback_detection_threshold를 맞추세요.'
```
<br>

이렇게 얻어진 threshold와 원하는 criteria를 config에 설정하면 ood 데이터셋 없이도
FallbackDetector를 이용할 수 있습니다. 그러나 지금 버전에서는 가급적 OOD 데이터셋을 이용해주세요. 
(정 없으시면 예제 데이터라도...)

```python
INTENT = {
    'distance_fallback_detection_criteria': 'mean', # or 'min'  
    # [auto, min, mean], auto는 OOD 데이터 있을때만 가능
    'distance_fallback_detection_threshold': 3.2,  
    # mean 혹은 min 선택시 임계값
    
    'softmax_fallback_detection_criteria': 'other',  
    # [auto, other], auto는 OOD 데이터 있을때만 가능
    'softmax_fallback_detection_threshold': 0.88,  
    # other 선택시 fallback이 되지 않는 최소 값
}
```
<br><br>


### 4.4. `from kochat.loss`
`loss` 패키지는 사전 정의된 다양한 built-in Loss 함수들이 저장된 패키지입니다.
현재 버전에서는 아래 목록에 해당하는 Loss 함수들을 지원합니다. 추후 버전이 업데이트 되면
지금보다 훨씬 다양한 built-in Loss 함수를 지원할 예정입니다. 아래 목록을 참고하여 사용해주시길 바랍니다.

<br>

#### 4.4.1. intent loss 함수
Intent Loss 함수는 기본적인 CrossEntropyLoss와 다양한 Distance 기반의 Loss함수를
활용할 수 있습니다. CrossEntropy는 후술할 Softmax 기반의 IntentClassifier에 주로
활용하고, Distance 기반의 Loss 함수들은 Distance 기반의 IntentClassifier에 
활용할 수 있습니다. Distance 기반의 Loss함수들은 컴퓨터 비전 영역 (주로 얼굴인식)
분야에서 제안된 함수들이지만 Intent 분류의 Fallback 디텍션에도 매우 우수한 성능을 보입니다.

<br>

```python
from kochat.loss import CrossEntropyLoss
from kochat.loss import CenterLoss
from kochat.loss import GaussianMixture
from kochat.loss import COCOLoss
from kochat.loss import CosFace


# 1. 가장 기본적인 Cross Entropy Loss 함수입니다.
cross_entropy = CrossEntropyLoss(label_dict=dataset.intent_dict)

# 2. Intra Class 간의 거리를 좁힐 수 있는 Center Loss 함수입니다.
center_loss = CenterLoss(label_dict=dataset.intent_dict)

# 3. Intra Class 간의 거리를 좁힐 수 있는 Large Margin Gaussian Mixture Loss 함수입니다.
lmgl = GaussianMixture(label_dict=dataset.intent_dict)

# 4. Inter Class 간의 Cosine 마진을 키울 수 있는 COCO (Congenerous Cosine) Loss 함수입니다.
coco_loss = COCOLoss(label_dict=dataset.intent_dict)

# 5. Inter Class 간의 Cosine 마진을 키울 수 있는 Cosface (Large Margin Cosine) Loss함수입니다.
cosface = CosFace(label_dict=dataset.intent_dict)
```
<br>

#### 4.4.2. entity loss 함수
Entity Loss 함수는 기본적인 CrossEntropyLoss와 확률적 모델인
Conditional Random Field (이하 CRF) Loss를 지원합니다.
CRF Loss를 적용하면, EntityRecognizer의 출력 결과를 다시한번 교정하는
효과를 볼 수 있으며 CRF Loss를 적용하면, 출력 디코딩은 Viterbi 알고리즘을 
통해 수행합니다.
<br>

```python
from kochat.loss import CrossEntropyLoss
from kochat.loss import CRFLoss


# 1. 가장 기본적인 cross entropy 로스 함수입니다.
cross_entropy = CrossEntropyLoss(label_dict=dataset.intent_dict)

# 2. CRF Loss 함수입니다.
center_loss = CRFLoss(label_dict=dataset.intent_dict)
```
<br>

#### 4.4.3. 커스텀 loss 함수
Kochat은 커스텀 모델을 지원합니다. 
Pytorch로 작성한 커스텀 모델을 직접 학습시키기고 챗봇 애플리케이션에 
사용할 수 있습니다. 그러나 만약 커스텀 모델을 사용하려면 아래의 몇가지 규칙을 반드시 
따라야합니다.
<br><br>

1. forward 함수에서 해당 loss를 계산합니다.
2. compute_loss 함수에서 라벨과 비교하여 최종 loss값을 계산합니다.
이 때 위에서 구현한 forward 함수를 활용합니다.
<br><br>

아래의 구현 예제를 보면 더욱 쉽게 이해할 수 있습니다.

```python
@intent
class CosFace(BaseLoss):

    def __init__(self, label_dict: dict):
        super(CosFace, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))

    def forward(self, feat: Tensor, label: Tensor) -> Tensor:
        # 1. forward 함수에서 전체 loss를 계산합니다.

        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.cosface_m)
        margin_logits = self.cosface_s * (logits - y_onehot)
        return margin_logits

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
        # 2. compute loss에서 최종 loss값을 계산합니다.

        mlogits = self(feats, label) # forward 호출
        return F.cross_entropy(mlogits, label)
```
```python
@intent
class CenterLoss(BaseLoss):
    def __init__(self, label_dict: dict):
        super(CenterLoss, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))
        self.center_loss_function = CenterLossFunction.apply

    def forward(self, feat: Tensor, label: Tensor) -> Tensor:
        # 1. forward 함수에서 전체 loss를 계산합니다.

        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()

        if feat.size(1) != self.d_loss:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}"
                             .format(self.d_loss, feat.size(1)))

        return self.center_loss_function(feat, label, self.centers)

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
        # 2. compute loss에서 최종 loss값을 계산합니다.

        nll_loss = F.cross_entropy(logits, label)
        center_loss = self(feats, label)  # forward 호출 
        return nll_loss + self.center_factor * center_loss
```
<br><br><br>

### 4.5. `from kochat.app`
`app` 패키지는 kochat 모델을 애플리케이션으로 배포할 수 있게끔 해주는 RESTful API은 `KochatApi`
클래스와 API 호출에 관련된 시나리오를 작성할 수 있게끔 하는 `Scenario`클래스를 제공합니다.

<br>

#### 4.5.1 `from kochat.app import Scenario`
`Scenario` 클래스는 어떤 intent에서는 어떤 entity가 필요하고, 
어떤 api를 호출하는지 정의하는 일종의 명세서와 같습니다. 
시나리오 작성시 아래와 같은 몇가지 주의사항이 있습니다.

1. intent는 반드시 raw데이터 파일 명과 동일하게 설정하기
2. api는 함수 그 자체를 넣습니다 (반드시 callable 해야합니다.)
3. scenario_dict 정의시에 KEY값은 api 함수와 순서/철자가 동일해야합니다.
4. 기본값(default) 설정을 원하면 scenario_dict 리스트에 값을 첨가합니다.
<br><br>

- kocrawl (날씨) 예제
```python
from kochat.app import Scenario
from kocrawl.weather import WeatherCrawler

# kocrawl은 kochat을 만들면서 함께 개발한 크롤러입니다.
# (https://github.com/gusdnd852/kocrawl)
# pip install kocrawl로 손쉽게 설치할 수 있습니다.


weather_scenario = Scenario(
    intent='weather',  # intent는 인텐트 명을 적습니다 (raw 데이터 파일명과 동일해야합니다)
    api=WeatherCrawler().request, # API는 함수 이름 자체를 넣습니다. (callable해야합니다)

    scenario_dict={
        'LOCATION': [],
        # 기본적으로 'KEY' : []의 형태로 만듭니다.

        'DATE': ['오늘']        
        # entity가 검출되지 않았을 때 default 값을 지정하고 싶으면 리스트 안에 원하는 값을 넣습니다.
        # [전주, 날씨, 알려줘] => [S-LOCATION, O, O] => api('오늘', S-LOCATION) call
        # 만약 ['오늘', '현재']처럼 2개 이상 넣으면 랜덤으로 선택해서 default 값으로 지정합니다.
    }

    # 시나리오 딕셔너리를 정의합니다.
    # 주의점 1 : scenario_dict 키값(LOCATION, DATE)의 순서는 API 함수의 파라미터 순서와 동일해야합니다.
    # 주의점 2 : scenario_dict 키값(LOCATION, DATE)의 철자는 API 함수의 파라미터 철자와 동일해야합니다.
    # 대/소문자까지 동일할 필요는 없습니다. 정확한 값 전달을 위해 일부러 만든 두 가지 제한사항입니다.
    
    # WeatherCrawler().request의 파라미터는 WeatherCrawler().request(location, date)입니다.
    # 주의점 1인 순서와 주의점 2인 철자가 모두 동일합니다. (만약 틀리면 어디서 틀렸는지 에러로 알려드립니다) 
)      
```

<br>

- 레스토랑 예약 시나리오
```python
from kochat.app import Scenario


reservation_scenario = Scenario(
    intent='reservation',
    api=reservation_check, 
    # reservation_check(num_people, reservation_time)와 같은
    # 함수를 호출하지 말고 그 자체를 파라미터로 입력합니다.
    # 함수를 받아서 저장해뒀다가 요청 발생시 Api 내부에서 call 합니다
    
    scenario_dict={
        'NUM_PEOPLE': [4],
        'RESERVATION_TIME': []

        # 파라미터와 순서/철자가 일치합니다.
        # NUM_PEOPLE의 default를 4명으로 설정했습니다.
    }
)     
```
<br><br>


## 5. Visualization Support

## 6. Demo Application

## 7. Performance Issue

## 8. Contributor

## 9. TODO List
- [x] ver 1.0 : 엔티티 학습에 CRF 및 로스 마스킹 추가하기 
- [x] ver 1.0 : 상세한 README 문서 작성 및 PyPI 배포하기
- [x] ver 1.0 : 간단한 웹 인터페이스 기반 데모 애플리케이션 제작하기
- [ ] ver 1.0 : Jupyter Note Example 작성하기 + Binder 실행 환경
- [ ] ver 1.1 : 데이터셋 포맷 RASA처럼 markdown에 대괄호 형태로 변경
- [ ] ver 1.2 : Transformer 기반 모델 추가 (SK KOBERT)
- [ ] ver 1.3 : Pytorch Embedding 모델 추가 (CharCNN + ELMO 추가)
- [ ] ver 1.4 : Seq2Seq 추가해서 Fallback시 대처할 수 있게 만들기 (SK KOGPT2)
- [ ] ver 1.5 : 네이버 맞춤법 검사기 제거하고, 자체적인 띄어쓰기 검사모듈 추가
- [ ] ver 1.5 : BERT와 Markov 체인을 이용한 자동 OOD 데이터 생성기능 추가
- [ ] ver 1.7 : 대화 흐름관리를 위한 Story 관리 기능 구현해서 추가하기
<br><br><br>

## 9. License
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
