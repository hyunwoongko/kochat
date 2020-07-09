
## 5. Visualization Support
Kochat은 아래와 같이 다양한 시각화 기능을 지원합니다.
Feature Space는 일정 Epoch마다 메모리에 저장되고,
그 외의 시각화 자료는 매 Epoch마다 계속 업데이트 되며
"root/saved"에 모델 저장파일과 함께 저장됩니다.
시각화 자료 및 모델 저장 경로는 
config에서 변경할 수 있습니다.
<br><br>

#### 5.1. Train/Test Accuracy

![](https://user-images.githubusercontent.com/38183241/86322540-2455ff80-bc76-11ea-9cb6-ec6eb196b89b.png)

<br><br>

#### 5.2. Train/Test Recall (macro average)
![](https://user-images.githubusercontent.com/38183241/86322532-21f3a580-bc76-11ea-9ef7-accf6ae7db19.png)

<br><br>

#### 5.3. Train/Test Precision (macro average)
![](https://user-images.githubusercontent.com/38183241/86322531-215b0f00-bc76-11ea-8844-1ee812e64c74.png)

<br><br>

#### 5.4. Train/Test F1-Score (macro average)
![](https://user-images.githubusercontent.com/38183241/86322529-2029e200-bc76-11ea-9163-30934b8bc5ff.png)

<br><br>

#### 5.5. Train/Test Confusion Matrix

![](https://user-images.githubusercontent.com/38183241/86396855-b47b5f80-bcdd-11ea-9672-4adf0f0ed140.png)

Confusion Matrix의 경우는 X축(아래)가 Prediction, Y축(왼쪽)이 Label입니다. 
<br>다음 버전에서 xticks와 yticks를 추가할 예정입니다.

<br><br>

#### 5.6. Train/Test Classification Performance Report

Accuracy, Precision, Recall, F1 Score 등 모델을 다양한 메트릭으로 평가하고,
표 형태로 이미지파일을 만들어줍니다.

![](https://user-images.githubusercontent.com/38183241/86397411-b8f44800-bcde-11ea-8b66-22423c12584c.png)

<br>

소수점 몇번째 까지 반올림해서 보여줄지 config에서 설정할 수 있습니다.
```python
PROC = {
    # ...(생략)
    'logging_precision': 5,  # 결과 저장시 반올림 소수점 n번째에서 반올림
}
```

<br><br>

#### 5.7. Train/Test Fallback Detection Performance Report

Fallback Detection은 Intent Classification의 영역입니다. Intent Classification만 지원합니다.
(Fallback Detection 성능 평가를 위해서는 반드시 ood=True여야합니다.)

![](https://user-images.githubusercontent.com/38183241/86323442-d17d4780-bc77-11ea-8c15-8be1eb4fa6e5.png)

<br><br>

#### 5.8. Feature Space Visualization

Feature Space는 Distance 기반의 Metric Learning Loss함수가 잘 작동하고 있는지
확인하기 위한것으로 Intent Classification만 지원합니다.
또한 시각화 차원은 config의 d_loss에 따라 결정됩니다.

- d_loss = 2인 경우 : 2차원으로 시각화
- d_loss = 3인 경우 : 3차원으로 시각화
- d_loss > 3인 경우 : Incremetal PCA를 통해 3차원으로 차원 감소 후 시각화
<br>

![](https://user-images.githubusercontent.com/38183241/86323429-c62a1c00-bc77-11ea-9caf-ede65f4cbc6c.png)
<br><br>

Feature Space Visualization은 PCA를 실행하기 때문에 비용이 상당히 큽니다.
다른 시각화는 매 Epoch마다 수행하지만, Feature Space Visulization은 몇 Epoch마다 
수행할지 결정할 수 있습니다. 

```python
PROC = {
    # ...(생략)
    'visualization_epoch': 50,  # 시각화 빈도 (애폭마다 시각화 수행)
}
```