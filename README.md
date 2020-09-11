# Kochat
[![PyPI version](https://badge.fury.io/py/kochat.svg)](https://badge.fury.io/py/kochat)
![GitHub](https://img.shields.io/github/license/gusdnd852/kochat)

![introduction_kochat](https://user-images.githubusercontent.com/38183241/85958000-1b8ed080-b9cd-11ea-99d6-69b472f3e2ff.jpg)
<br>

![](https://user-images.githubusercontent.com/38183241/86410173-4347a680-bcf5-11ea-9261-e272ad21ed36.gif)
<br><br>

- 챗봇 빌더는 성에 안차고, 자신만의 딥러닝 챗봇 애플리케이션을 만드시고 싶으신가요?
- Kochat을 이용하면 손쉽게 자신만의 딥러닝 챗봇 애플리케이션을 빌드할 수 있습니다.

```python
# 1. 데이터셋 객체 생성
dataset = Dataset(ood=True)

# 2. 임베딩 프로세서 생성
emb = GensimEmbedder(model=embed.FastText())

# 3. 의도(Intent) 분류기 생성
clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),                  
    loss=CenterLoss(dataset.intent_dict)                    
)

# 4. 개체명(Named Entity) 인식기 생성                                                     
rcn = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

# 5. 딥러닝 챗봇 RESTful API 학습 & 빌드
kochat = KochatApi(
    dataset=dataset, 
    embed_processor=(emb, True), 
    intent_classifier=(clf, True),
    entity_recognizer=(rcn, True), 
    scenarios=[
        weather, dust, travel, restaurant
    ]
)

# 6. View 소스파일과 연결                                                                                                        
@kochat.app.route('/')
def index():
    return render_template("index.html")

# 7. 챗봇 애플리케이션 서버 가동                                                          
if __name__ == '__main__':
    kochat.app.template_folder = kochat.root_dir + 'templates'
    kochat.app.static_folder = kochat.root_dir + 'static'
    kochat.app.run(port=8080, host='0.0.0.0')
```
<br>

### Warning
현재 버전은 GPU에서 KNN의 속도가 너무 느리다는 이슈가 발견 되어
문제를 해결했으나, 아직 완성되지 않아서 pip로는 배포하지 않았으니,
다소 학습속도가 느리더라도 pip에 배포된 버전(1.0.3)과 아래의
공식 도큐먼테이션에 나온 코드를 이용해주세요.

<br><br>

## Why Kochat?
- 한국어를 지원하는 최초의 오픈소스 딥러닝 챗봇 프레임워크입니다. (빌더와는 다릅니다.)
- 다양한 Pre built-in 모델과 Loss함수를 지원합니다. NLP를 잘 몰라도 챗봇을 만들 수 있습니다.
- 자신만의 커스텀 모델, Loss함수를 적용할 수 있습니다. NLP 전문가에겐 더욱 유용합니다.
- 챗봇에 필요한 데이터 전처리, 모델, 학습 파이프라인, RESTful API까지 모든 부분을 제공합니다.
- 가격 등을 신경쓸 필요 없으며, 앞으로도 쭉 오픈소스 프로젝트로 제공할 예정입니다.
- 아래와 같은 다양한 성능 평가 메트릭과 강력한 시각화 기능을 제공합니다.

![](https://user-images.githubusercontent.com/38183241/86397184-513dfd00-bcde-11ea-9540-aa56a24b6d9b.png)

![](https://user-images.githubusercontent.com/38183241/86397411-b8f44800-bcde-11ea-8b66-22423c12584c.png)

![](https://user-images.githubusercontent.com/38183241/86396855-b47b5f80-bcdd-11ea-9672-4adf0f0ed140.png)

![](https://user-images.githubusercontent.com/38183241/86323429-c62a1c00-bc77-11ea-9caf-ede65f4cbc6c.png)
<br><br><br>

## Documentation

1. [Kochat이란?](https://github.com/gusdnd852/kochat/tree/master/docs/01_kocaht_이란.md)
2. [About Chatbot](https://github.com/gusdnd852/kochat/tree/master/docs/02_about_chatbot.md)
3. [Getting Started](https://github.com/gusdnd852/kochat/tree/master/docs/03_getting_started.md)
4. [Usage](https://github.com/gusdnd852/kochat/tree/master/docs/04_usage.md)
5. [Visualization Support](https://github.com/gusdnd852/kochat/tree/master/docs/05_visualization_support.md)
6. [Performance Issue](https://github.com/gusdnd852/kochat/tree/master/docs/06_performance_issue.md)
7. [Demo](https://github.com/gusdnd852/kochat/tree/master/docs/07_demo.md)


<br>


## TODO List
- [x] ver 1.0 : 엔티티 학습에 CRF 및 로스 마스킹 추가하기 
- [x] ver 1.0 : 상세한 README 문서 작성 및 PyPI 배포하기
- [x] ver 1.0 : 간단한 웹 인터페이스 기반 데모 애플리케이션 제작하기
- [ ] ver 1.0 : Jupyter Note Example 작성하기 + Colab 실행 환경
- [x] ver 1.0 : 데모 데이터셋 10배로 확장 (intent 당 600 → 5000 라인)
- [ ] ver 1.1 : OOD 데이터셋 없이 OOD 분류기능 학습 (구현완료, 마무리필요)
- [ ] ver 1.2 : 데이터셋 포맷 RASA처럼 markdown에 대괄호 형태로 변경
- [ ] ver 1.2 : System Entity & Intent 데이터셋 지원 (일단 NER 데이터 긁어모으고 인텐트는 음..)
- [ ] ver 1.3 : Pretrain Embedding 적용 가능하게 변경 (Gensim)
- [ ] ver 1.4 : Transformer 기반 모델 추가 (Etri BERT, SK BERT) with Hugging face
- [ ] ver 1.5 : Pytorch Embedding 모델 추가 + Pretrain 적용 가능하게
- [ ] ver 1.4 : Intent & Entity 멀티모달 지원 (학습속도 및 추론 개선을 위함)
- [ ] ver 1.7 : Seq2Seq 추가해서 Fallback시 대처할 수 있게 만들기 (LSTM, SK GPT2)
- [ ] ver 1.8 : 네이버 맞춤법 검사기 제거하고, 자체적인 띄어쓰기 검사모듈 추가
- [ ] ver 1.9 : 대화 흐름관리를 위한 Story 관리 기능 구현해서 추가하기
<br><br><br>

## Reference
- [챗봇 분류 그림](https://towardsdatascience.com/chatbots-are-cool-a-framework-using-python-part-1-overview-7c69af7a7439)
- [seq2seq 그림](https://mc.ai/implement-of-seq2seq-model/)
- [Fallback Detection 그림](https://docs.smartly.ai/docs/intent-detection)
- [데모 애플리케이션 템플릿](https://bootsnipp.com/snippets/ZlkBn)
- 그 외의 그림 및 소스코드 : 본인 제작
<br><br><br>

## License
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
