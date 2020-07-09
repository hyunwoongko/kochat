
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
[kochat_config.zip](https://github.com/gusdnd852/kochat/files/4867232/kochat_config.zip) 을 
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
파주 유명한 공연장 알려줘,S-LOCATION O S-PLACE O
창원 여행 갈만한 바다,S-LOCATION O O S-PLACE
평택 갈만한 스키장 여행 해보고 싶네,S-LOCATION O S-PLACE O O O
제주도 템플스테이 여행 갈 데 추천해 줘,S-LOCATION S-PLACE O O O O O
전주 가까운 바다 관광지 보여줘 봐요,S-LOCATION O S-PLACE O O O
용인 가까운 축구장 어딨어,S-LOCATION O S-PLACE O
붐비는 관광지,O O
청주 가을 풍경 예쁜 산 가보고 싶어,S-LOCATION S-DATE O O S-PLACE O O
... (생략)
```

위 처럼 question,label이라는 헤더(컬럼명)을 가장 윗줄에 위치시키고,
그 아래로 두개의 컬림 question과 label에 해당하는 내용을 작성합니다.
각 단어 및 엔티티는 띄어쓰기로 구분됩니다.
데모 데이터는 BIO태깅을 개선한 BIOES태깅을 사용하여 라벨링했는데, 엔티티 태깅 방식은 자유롭게
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

#### 3.5.5. 라벨링 실수 검출
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

    'NER_categories': ['DATE', 'LOCATION', 'RESTAURANT', 'PLACE'],  # 사용자 정의 태그
    'NER_tagging': ['B', 'E', 'I', 'S'],  # NER의 BEGIN, END, INSIDE, SINGLE 태그
    'NER_outside': 'O',  # NER의 O태그 (Outside를 의미)
}

question = 전주 눈 올까
label = Z-LOC O O

→ 에러 발생! (정의되지 않은 엔티티 : Z-LOC)
NER_tagging + '-' + NER_categories의 형태가 아니면 에러를 반환합니다.
```
<br>

#### 3.5.6. OOD 데이터셋
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
이 의도값을 사용하진 않기 때문에 그냥 아무값으로나 라벨링해도 사실 무관합니다.

```
데모_ood_데이터.csv

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

데이터까지 모두 삽입하셨다면 kochat을 이용할 준비가 끝났습니다. 
이제 다음 챕터에서는 자세한 사용법에 대해 알려드리겠습니다.
<br><br><br>
