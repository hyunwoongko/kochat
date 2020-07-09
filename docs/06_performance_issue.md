
## 6. Performance Issue
이 챕터는 Kochat의 다양한 성능 이슈에 대해 기록합니다.

<br><br>

#### 6.1. 얼굴인식 영역에서 쓰이던 Loss 함수들은 Fallback 디텍션에 효과적이다.
사실 CenterLoss나 CosFace 같은 Margin Loss함수들이 컴퓨터 비전의 얼굴인식 영역에서 
많이 쓰인다고는 하나 기본적으로 모든 Retrieval 문제에 적용할 수 있는 Loss함수입니다.
Kochat의 DistanceClassifier는 거리기반의 Retrieval을 수행하기 때문에 이러한
Loss함수를 매우 효과적으로 활용할 수 있습니다. 실제로 데모 데이터셋에 적용했을 때
CrossEntropyLoss로는 70% 언저리인 FallbackDetection 성능이 CenterLoss, CosFace 
등을 적용하면 90~95%까지 향상되었습니다. (120개의 OOD 샘플 테스트)
<br><br>

- SoftmaxClassifier + CrossEntropyLoss + CNN (d_model=512, layers=1)

![](https://user-images.githubusercontent.com/38183241/86393797-834c6080-bcd8-11ea-86f0-3fc4c897382d.png)

<br>

- DistanceClassifier + CrossEntropyLoss + CNN (d_model=512, layers=1)

![](https://user-images.githubusercontent.com/38183241/86393467-1638cb00-bcd8-11ea-8d04-d663ce89d124.png)

<br>

- DistanceClassifier + CenterLoss + CNN (d_model=512, layers=1)

![](https://user-images.githubusercontent.com/38183241/86323442-d17d4780-bc77-11ea-8c15-8be1eb4fa6e5.png)

<br>


#### 6.2. Retrieval Feature로는 LSTM보다 CNN이 더 좋다.
Retrieval 기반의 Distance Classification의 경우 LSTM보다 CNN의 Feature들이 
클래스별로 훨씬 잘 구분되는 것을 확인했습니다. Feature Extraction 능력 자체는 
CNN이 좋다고 알려진 것처럼 아무래도 CNN이 Feature를 더 잘 뽑아내는 것 같습니다.
Feature Space에서 구분이 잘 된다는 것은 OOD 성능이 우수하다는 것과 동치이므로, 
DistanceClassifier 사용시 LSTM보단 CNN을 사용하는 것이 더욱 바람직해보입니다.
<br><br>

- 좌 : LSTM (d_model=512, layers=1) + CosFace, 500 Epoch 학습 (수렴함)
- 우 : CNN (d_model=512, layers=1) + CosFace, 500 Epoch 학습 (수렴함)

![image](https://user-images.githubusercontent.com/38183241/86394150-0ff71e80-bcd9-11ea-97c8-e0939b8f3f5d.png)

<br><br>

#### 6.3. CRF Loss의 수렴 속도는 CrossEntropy보다 느리다.

EntityRecognizer의 경우 동일 사이즈, 동일 Layer에서 CRF Loss를 사용하면
확실히 성능은 더욱 우수해지나, 조금 더 더 느리게 수렴하는 것을 확인했습니다. 
CRF Loss의 경우 조금 더 많은 학습 시간을 줘야 제 성능을 내는 것 같습니다.
<br><br>

- 좌 : LSTM (d_model=512, layers=1) + CrossEntropy → Epoch 300에 f1-score 90% 도달
- 우 : LSTM (d_model=512, layers=1) + CRFLoss → Epoch 450에 f1-score 90% 도달

![](https://user-images.githubusercontent.com/38183241/86394923-4bdeb380-bcda-11ea-9d70-ec4da761893b.png)

<br><br>


#### 6.4. Sklearn은 GPU 가속이 안된다. 정말 느리다. (v 1.1 추가예정)
기존에 Sklearn의 KNN으로 KD 트리를 만들고 K개의 가장 가까운 피쳐벡터를 선정하는
방식으로 구현했습니다. 아무리 KD 트리를 쓴다고 해도, GPU 가속이 안되는 Sklearn으로
대용량 데이터를 다루기에는 역부족이였습니다. 기존처럼 K개의 벡터를 선정하는 방식으로
가려면 결국 n개의 데이터가 있을 때 n개를 모두 다 검색해야한다는 것인데, 그렇게 되면
Time complexity는 O(n)이 됩니다. 실제로 기존 데모데이터를 3000 line에서 
20000 line으로 늘리고 다시 테스트 했더니 ETA가 1분이 넘어갔습니다. <br><br>

그에 비해 SoftmaxClassifier는 동일한 데이터로 4 ~ 6초정도의 ETA를 보여줬습니다. 
즉 torch부분은 문제 없고 sklearn의 검색 속도에 문제가 있다는 것이 확실했습니다. 
이 문제를 해결하기 위해서 CenterLoss나 CosFace 등이 매 Epoch마다 계산하는 클래스들의 
중심점에 포커싱 해봤습니다. 모든 샘플을 전부 다 저장하고 검색할 것이 아니라, 
그 샘플들이 따라가고 있는 중심점이랑만 비교해도 성능이 보장 될 것이라는 판단이였습니다.
또한 기존에 Cosine Similarity 기반의 CosFace나 COCO Loss는 거리가 아닌 Cosine
Similarity 기반으로 비교하는 것이 올바르나, 무리하게 KNN을 사용했기 때문에 이들마저도 거리로
계산해왔습니다. 때문에 Retrieval 전략을 아래와 같이 변경하였습니다.
<br><br>


1. Distance 기반의 Loss (CenterLoss, GaussianMixture)는 중심점과의 거리 비교로 검색
2. Cosine Similarity 기반의 Loss (CosFace, COCOLoss)는 중심점과의 각도 비교로 검색
<br><br>


오로지 중심점과 비교한다는 전략은 매우 성공적이였고, Time complexity는
O(classes)이나, 데이터가 추가되어도 대부분의 경우 class의 갯수는 고정되기 때문에
사실상 O(1)의 Time complexity를 보장할 수 있게 되었습니다. 또한 기존에
제대로 평가하지 못했던 Cosine 기반의 Loss함수들도 KNN을 제거하면서 보다 정확한
메트릭을 사용하여 Retrieval 할 수 있게 되었습니다. 새로운 방식은 
기존과 동일한 성능을 보이면서 Softmax Classifier와 비슷한 속도를 가지게 되었습니다.
<br><br>
