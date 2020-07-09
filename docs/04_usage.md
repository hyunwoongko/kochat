
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
`SoftmaxClassifier`는 가장 기본적인 분류 프로세서입니다.
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
와의 성능 비교를 위해 구현하였습니다. 사용법은 아래와 같습니다.

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

# 모델 추론 (인텐트 분류)
clf.predict(dataset.load_predict("오늘 서울 날씨 어떨까", emb))
```

<br>

#### 4.3.3. `from kochat.proc import DistanceClassifier`
`DistanceClassifier`는 `SoftmaxClassifier`와는 다르게 거리기반으로 작동하며,
일종의 Memory Network입니다. [batch_size, -1] 의 사이즈로 출력된 출력벡터와 
기존 데이터셋에 있는 문장 벡터들 사이의 거리를 계산하여 데이터셋에서 가장 가까운 
K개의 샘플을 찾고 최다 샘플 클래스로 분류하는 최근접 이웃 Retrieval 기반의 분류 모델입니다.
<br><br>

이 때 다른 클래스들은 멀리, 같은 클래스끼리는 가까이 있어야
분류하기에 좋기 때문에 사용자가 설정한 Loss함수(주로 Margin 기반 Loss)를 
적용해 Metric Learning을 수행해서 클래스 간의 Margin을 최대치로 벌리는 
메커니즘이 구현되어있습니다. 또한 최근접 이웃 알고리즘의 K값은 config에서 
직접 지정 할 수도 있고 GridSearch를 적용하여 자동으로 최적의 K값을 찾을 수 있게 설계하였습니다. 

<br>

최근접 이웃을 찾을 때 Brute force로 직접 거리를 일일이 다 구하면 굉장히 느리기 
때문에 다차원 검색트리인 `KDTree` 혹은 `BallTree` (KDTree의 개선 형태)를 통해서 
거리를 계산하며 결과로 만들어진 트리 구조를 메모리에 저장합니다. 검색트리의 종류, 
거리 메트릭(유클리디언, 맨하튼 등..)은 전부 GridSearch로 자동화 시킬 수 있으며
이에 대한 설정은 config에서 가능합니다. 사용법은 아래와 같습니다.
(대량의 데이터셋을 이용하면 속도가 매우 느린 것으로 확인 되었습니다.)

```python
from kochat.data import Dataset
from kochat.proc import BaseClassifier
from kochat.model import intent
from kochat.loss import CenterLoss


dataset = Dataset(ood=True)

# 프로세서 생성
clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict)
)

# 되도록이면 DistanceClassifier는 Margin 기반의 Loss 함수를 이용해주세요
# 현재는 CenterLoss, COCOLoss, Cosface, GausianMixture 등의 
# 거리기반 Metric Learning 전용 Loss함수를 지원합니다.


# 모델 학습
clf.fit(dataset.load_intent(emb))

# 모델 추론 (인텐트 분류)
clf.predict(dataset.load_predict("오늘 서울 날씨 어떨까", emb))
```
<br><br>

#### 4.3.4. `FallbackDetector`
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
        LogisticRegression(max_iter=30000),
        LinearSVC(max_iter=30000)

        # 가능한 max_iter를 높게 설정해주세요
        # sklearn default가 max_iter=100이라서 수렴이 안됩니다...
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
안정적인 성능을 보였습니다.

<br>

이렇게 Fallback의 메트릭으로 머신러닝 모델을 학습하면 Threshold를 직접 유저가 
설정하지 않아도 됩니다. OOD 데이터셋이 필요하다는 치명적인 단점이 있지만, 
차후 버전에서는 BERT와 Markov Chain을 이용해 OOD 데이터셋을 자동으로 빠르게 생성하는 
모델을 구현하여 추가할 예정입니다. (이 업데이트 이후부터는 OOD 데이터셋이 필요 없어집니다.)

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

이렇게 calibrate 모드를 여러번 진행하셔서 스스로 계산한 threshold와 원하는 criteria를 아래처럼 
config에 설정하면 ood 데이터셋 없이도 FallbackDetector를 이용할 수 있습니다. 

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
<br>

그러나 지금 버전에서는 가급적 OOD 데이터셋을 추가해서 이용해주세요. 
정 없으시면 제가 데모 폴더에 넣어놓은 데이터라도 넣어서 자동화해서 쓰는게 
훨씬 성능이 좋습니다. 몇몇 빌더들은 이 임계치를 직접 정하게 하거나 그냥 상수로 
fix해놓는데, 개인적으로 이걸 그냥 상수로 fix 해놓거나 유저보고 직접 정하게 하는건 
챗봇 빌더로서, 혹은 프레임워크로서 무책임한 것 아닌가 싶습니다. 
<br><br><br>

#### 4.3.5. `from kochat.proc import EntityRecongnizer`
`EntityRecongnizer`는 엔티티 검출을 담당하는 Entity 모델들을 학습/테스트 시키고 추론하는 
클래스입니다. Entity 검사의 경우 문장 1개당 라벨이 여러개(단어 갯수와 동일)입니다.
문제는 Outside 토큰인 'O'가 대부분이기 때문에 전부다 'O'라고만 예측해도 거의 90% 육박하는
정확도가 나오게 됩니다. 또한, 패드시퀀싱한 부분도 'O'로 처리 되어있는데, 이 부분도 맞은것으로
생각하고 Loss를 계산합니다. 

<br>

이러한 문제를 해결하기 위해 Kochat은 F1 Score, Recall, Precision 등 
NER의 성능을 보다 정확하게 평가 할 수 있는 강력한 Validation 및 시각화 지원과 
Loss 함수 계산시 PAD부분에 masking을 적용할 수 있습니다. 
(mask 적용 여부 역시 config에서 설정 가능합니다.)
사용법은 아래와 같습니다.


```python
from kochat.data import Dataset
from kochat.proc import EntityLSTM
from kochat.model import entity
from kochat.loss import CRFLoss


dataset = Dataset(ood=True)

# 프로세서 생성
rcn = EntityRecognizer(
    model=entity.LSTM(dataset.intent_dict),
    loss=CRFLoss(dataset.intent_dict)
    # Conditional Random Field를 Loss함수로 지원합니다.
)


# 모델 학습
rcn.fit(dataset.load_entity(emb))

# 모델 추론 (엔티티 검출)
rcn.predict(dataset.load_predict("오늘 서울 날씨 어떨까", emb))
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
        # 1. forward 함수에서 현재 loss함수의 loss를 계산합니다.

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

        mlogits = self(feats, label)
        # 자기 자신의 forward 호출
        
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
        # 1. forward 함수에서 현재 loss함수의 loss를 계산합니다.

        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()

        if feat.size(1) != self.d_loss:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}"
                             .format(self.d_loss, feat.size(1)))

        return self.center_loss_function(feat, label, self.centers)

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
        # 2. compute loss에서 최종 loss값을 계산합니다.

        nll_loss = F.cross_entropy(logits, label)
        center_loss = self(feats, label)
        # 자기 자신의 forward 호출

        return nll_loss + self.center_factor * center_loss
```
<br><br><br>

### 4.5. `from kochat.app`
`app` 패키지는 kochat 모델을 애플리케이션으로 배포할 수 있게끔 해주는 
RESTful API인 `KochatApi`클래스와 API 호출에 관련된 시나리오를 
작성할 수 있게끔 하는 `Scenario`클래스를 제공합니다.

<br>

#### 4.5.1 `from kochat.app import Scenario`
`Scenario` 클래스는 어떤 intent에서는 어떤 entity가 필요하고, 
어떤 api를 호출하는지 정의하는 일종의 명세서와 같습니다. 
시나리오 작성시 아래와 같은 몇가지 주의사항이 있습니다.

1. intent는 반드시 raw데이터 파일 명과 동일하게 설정하기
2. api는 함수 그 자체를 넣습니다 (반드시 callable 해야합니다.)
3. scenario 딕셔너리 정의시에 KEY값은 api 함수와 순서/철자가 동일해야합니다.
4. scenario 딕셔너리 정의시에 KEY값은 config의 NER_categories에 정의된 엔티티만 허용됩니다.
4. 기본값(default) 설정을 원하면 scenario 딕셔너리의 리스트에 값을 첨가합니다.
<br><br>

- kocrawl (날씨) 예제
```python
from kochat.app import Scenario
from kocrawl.weather import WeatherCrawler

# kocrawl은 kochat을 만들면서 함께 개발한 크롤러입니다.
# (https://github.com/gusdnd852/kocrawl)
# 'pip install kocrawl'로 손쉽게 설치할 수 있습니다.


weather_scenario = Scenario(
    intent='weather',  # intent는 인텐트 명을 적습니다 (raw 데이터 파일명과 동일해야합니다)
    api=WeatherCrawler().request, # API는 함수 이름 자체를 넣습니다. (callable해야합니다)

    scenario={
        'LOCATION': [],
        # 기본적으로 'KEY' : []의 형태로 만듭니다.

        'DATE': ['오늘']        
        # entity가 검출되지 않았을 때 default 값을 지정하고 싶으면 리스트 안에 원하는 값을 넣습니다.
        # [전주, 날씨, 알려줘] => [S-LOCATION, O, O] => api('오늘', S-LOCATION) call
        # 만약 ['오늘', '현재']처럼 2개 이상의 default를 넣으면 랜덤으로 선택해서 default 값으로 지정합니다.
    }

    # 시나리오 딕셔너리를 정의합니다.
    # 주의점 1 : scenario 키값(LOCATION, DATE)의 순서는 API 함수의 파라미터 순서와 동일해야합니다.
    # 주의점 2 : scenario 키값(LOCATION, DATE)의 철자는 API 함수의 파라미터 철자와 동일해야합니다.
    # 주의점 3 : raw 데이터 파일에 라벨링한 엔티티명과 scenario 키값은 동일해야합니다. 
    #           즉 config의 NER_categories에 미리 정의된 엔티티만 사용하셔야합니다.
    #           B-, I- 등의 BIO태그는 생략합니다. (S-DATE → DATE로 생각)

    # 대/소문자까지 동일할 필요는 없고, 철자만 같으면 됩니다. (모두 lowercase 상태에서 비교)
    # 다소 귀찮더라도 정확한 값 전달을 위해 일부러 만든 세 가지 제한사항이니 따라주시길 바랍니다.

    # WeatherCrawler().request의 파라미터는 WeatherCrawler().request(location, date)입니다.
    # API파라미터와 순서/이름이 동일하며, 데모 데이터 파일에 있는 엔티티인 LOCATION, DATE와 동일합니다.
    # 만약 틀리면 어디서 틀렸는지 에러 메시지로 알려드립니다.
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
    
    scenario={
        'NUM_PEOPLE': [4],
        # NUM_PEOPLE의 default를 4명으로 설정했습니다.

        'RESERVATION_TIME': []

        # API(reservation_check(num_people, reservation_time)의 파라미터와 순서/철자가 일치합니다.
        # 이 때, 반드시 NER_categories에 NUM_PEOPLE과 RESERVATION_TIME이 정의되어 있어야하며,
        # 실제 raw데이터에 라벨링된 레이블도 위의 이름을 사용해야합니다.
    }
)     
```
<br><br>

#### 4.5.2. `from kochat.app import KochatApi`
`KochatApi`는 Flask로 구현되었으며 restful api를 제공하는 클래스입니다.
사실 서버로 구동할 계획이라면 위에서 설명한 것 보다 훨씬 쉽게 학습할 수 있습니다. 
(학습의 많은 부분들이 `KochatApi`에서 자동화 되기 때문에 파라미터 전달만으로 학습이 가능합니다.)
`KochatApi` 클래스는 아래와 같은 메소드들을 지원하며 사용법은 다음과 같습니다.

```python
from kochat.app import KochatApi


# kochat api 객체를 생성합니다.
kochat = KochatApi(
    dataset=dataset, # 데이터셋 객체
    embed_processor=(emb, True), # 임베딩 프로세서, 학습여부
    intent_classifier=(clf, True), # 인텐트 분류기, 학습여부
    entity_recognizer=(rcn, True), # 엔티티 검출기, 학습여부
    scenarios=[ #시나리오 리스트
        weather, dust, travel, restaurant
    ]
)

# kochat.app은 FLask 객체입니다. 
# Flask의 사용법과 동일하게 사용하면 됩니다.
@kochat.app.route('/')
def index():
    return render_template("index.html")


# 애플리케이션 서버를 가동합니다.
if __name__ == '__main__':
    kochat.app.template_folder = kochat.root_dir + 'templates'
    kochat.app.static_folder = kochat.root_dir + 'static'
    kochat.app.run(port=8080, host='0.0.0.0')
```

<br>

위와 같이 kochat 서버를 실행시킬 수 있습니다. 
(웬만하면 위와 같이 template과 static을 명시적으로 적어주세요.)
위 예시처럼 뷰를 직접 서버에 연결해서 하나의 서버에서 뷰와 딥러닝 코드를 
모두 구동시킬 수도 있고, 만약 Micro Service Architecture를 구성해야한다면,
챗봇 서버의 index route ('/')등을 설정하지 않고 딥러닝 백엔드 서버로도
충분히 활용할 수 있습니다. 만약 학습을 원하지 않을 때는 아래처럼 구현합니다.

```python
# 1. Tuple의 두번째 인자에 False 입력
kochat = KochatApi(
    dataset=dataset, # 데이터셋 객체
    embed_processor=(emb, False), # 임베딩 프로세서, 학습여부
    intent_classifier=(clf, False), # 인텐트 분류기, 학습여부
    entity_recognizer=(rcn, False), # 엔티티 검출기, 학습여부
    scenarios=[ #시나리오 리스트
        weather, dust, travel, restaurant
    ]
)

# 2. Tuple에 프로세서만 입력
kochat = KochatApi(
    dataset=dataset, # 데이터셋 객체
    embed_processor=(emb), # 임베딩 프로세서
    intent_classifier=(clf), # 인텐트 분류기
    entity_recognizer=(rcn), # 엔티티 검출기
    scenarios=[ #시나리오 리스트
        weather, dust, travel, restaurant
    ]
)

# 3. 그냥 프로세서만 입력
kochat = KochatApi(
    dataset=dataset, # 데이터셋 객체
    embed_processor=emb, # 임베딩 프로세서
    intent_classifier=clf, # 인텐트 분류기
    entity_recognizer=rcn, # 엔티티 검출기
    scenarios=[ #시나리오 리스트
        weather, dust, travel, restaurant
    ]
)
```

<br>

아래에서는 Kochat 서버의 url 패턴에 대해 자세하게 설명합니다.
현재 kochat api는 다음과 같은 4개의 url 패턴을 지원하며,
이 url 패턴들은 config의 API 챕터에서 변경 가능합니다.

```python
API = {
    'request_chat_url_pattern': 'request_chat',  # request_chat 기능 url pattern
    'fill_slot_url_pattern': 'fill_slot',  # fill_slot 기능 url pattern
    'get_intent_url_pattern': 'get_intent',  # get_intent 기능 url pattern
    'get_entity_url_pattern': 'get_entity'  # get_entity 기능 url pattern
}
```

<br>

#### 4.5.2.1. request_chat
가장 기본적인 패턴인 request_chat입니다. intent분류, entity검출, api연결을 한번에 진행합니다.
<br>
기본 패턴 : https://0.0.0.0/request_chat/<uid>/<text>
```
case 1. state SUCCESS
모든 entity가 정상적으로 입력된 경우 state 'SUCCESS'를 반환합니다.

>>> 유저 gusdnd852 : 모레 부산 날씨 어때

https://123.456.789.000:1234/request_chat/gusdnd852/모레 부산 날씨 어때
→ {
    'input': [모레, 부산, 날씨, 어때],
    'intent': 'weather',
    'entity': [S-DATE, S-LOCATION, O, O]
    'state': 'SUCCESS',
    'answer': '부산의 날씨 정보를 전해드릴게요. 😉
               모레 부산지역은 오전에는 섭씨 19도이며, 아마 하늘이 맑을 것 같아요. 오후에도 섭씨 26도이며, 아마 하늘이 맑을 것 같아요.'
}


case 2. state REQUIRE_XXX
만약 default값이 없는 엔티티가 입력되지 않은 경우 state 'REQUIRE_XXX'를 반환합니다.
두개 이상의 엔티티가 모자라면 state 'REQUIRE_XXX_YYY'가 반환됩니다.

>>> 유저 minqukanq : 목요일 날씨 어때

e.g. https://123.456.789.000:1234/request_chat/minqukanq/목요일 날씨 어때
→ {
    'input': [목요일, 날씨, 어때],
    'intent': 'weather',
    'entity': [S-DATE, O, O]
    'state': 'REQUIRE_LOCATION',
    'answer': None
}


case 3. state FALLBACK
인텐트 분류시 FALLBACK이 발생하면 FALLBACK을 반환합니다.

>>> 유저 sangji11 : 목요일 친구 생일이다

e.g. https://123.456.789.000:1234/request_chat/sangji11/목요일 친구 생일이다
→ {
    'input': [목요일, 친구, 생일이다],
    'intent': 'FALLBACK',
    'entity': [S-DATE, O, O]
    'state': 'FALLBACK',
    'answer': None
}
```

<br>

#### 4.5.2.2. fill_slot
가장 request시 REQUIRE_XXX가 나올때, 사용자에게 되묻고 기존 딕셔너리에 추가해서 api를 호출합니다.
<br>
기본 패턴 : https://0.0.0.0/fill_slot/<uid>/<text>
```
>>> 유저 gusdnd852 : 모레 날씨 알려줘 → REQUIRE_LOCATION
>>> 봇 : 어느 지역을 알려드릴까요?
>>> 유저 gusdnd852 : 부산

https://123.456.789.000:1234/fill_slot/gusdnd852/부산
→ {
    'input': [부산] + [모레, 날씨, 어때],
    'intent': 'weather',
    'entity': [S-LOCATION] + [S-DATE, O, O]
    'state': 'SUCCESS',
    'answer': '부산의 날씨 정보를 전해드릴게요. 😉
               모레 부산지역은 오전에는 섭씨 19도이며, 아마 하늘이 맑을 것 같아요. 오후에도 섭씨 26도이며, 아마 하늘이 맑을 것 같아요.'
}


>>> 유저 gusdnd852 : 날씨 알려줘 → REQUIRE_DATE_LOCATION
>>> 봇 : 언제의 어느 지역을 날씨를 알려드릴까요?
>>> 유저 gusdnd852 : 부산 모레

https://123.456.789.000:1234/fill_slot/gusdnd852/부산 모레
→ {
    'input': [부산, 모레] + [날씨, 어때],
    'intent': 'weather',
    'entity': [S-LOCATION, S-DATE] + [O, O]
    'state': 'SUCCESS',
    'answer': '부산의 날씨 정보를 전해드릴게요. 😉
               모레 부산지역은 오전에는 섭씨 19도이며, 아마 하늘이 맑을 것 같아요. 오후에도 섭씨 26도이며, 아마 하늘이 맑을 것 같아요.'
}
```

<br>

#### 4.5.2.3. get_intent
intent만 알고싶을때 호출합니다.
<br>
기본 패턴 : https://0.0.0.0/get_intent/<text>
```

https://123.456.789.000:1234/get_intent/전주 날씨 어때
→ {
    'input': [전주, 날씨, 어때],
    'intent': 'weather',
    'entity': None,
    'state': 'REQUEST_INTENT',
    'answer': None
}
```

<br>

#### 4.5.2.4. get_entity
entity만 알고싶을때 호출합니다.
<br>
기본 패턴 : https://0.0.0.0/get_entity/<text>
```

https://123.456.789.000:1234/get_entity/전주 날씨 어때
→ {
    'input': [전주, 날씨, 어때],
    'intent': None,
    'entity': [S-LOCATION, O, O],
    'state': 'REQUEST_ENTITY',
    'answer': None
}
```
