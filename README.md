# Chatbot based on CNN & LSTM

This repository introduces a simple chatbot architecture based on CNN & LSTM. The architecture includes intent classification of conversations, entity recognition, and providing information through various API connection.
<br><br>
In addition, all data and example applications are in Korean, and if you want to run this application in another language, you can modify the csv file in the 'src/data' folder. 
Each file has example data for each intent, so if you want to use the new data, you can create a file for each conversation intent.
<br><br>

##  0. Development environment
* OS : Windows 10
* IDE : IntelliJ 2019.01
* Language : Python 3.6
* PC Specifications :
  * CPU : Intel(R) Core(TM) i7-9700KF @ 3.60Ghz
  * RAM : Samsung 16GB
  * GPU : Nvidia RTX 2070
<br><br><br>

## 1. Intent Classification

First, if the user tries to talk, we must know the intent of the conversation.
<br>
<br>
Because of this, Convolution neural networks are used to classify users intent. The way to classify sentences using the Convolution network is based on the [paper by Kim Yoon, published in 2014](https://www.aclweb.org/anthology/D14-1181).

<img src=https://i.imgur.com/TNjCKHf.jpg></img> <br><br>

The parameters used for training are as follows.

    encode_length = 15
    filter_sizes = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
    learning_step = 3001
    learning_rate = 0.00001
    vector_size = 300
    fallback_score = 10
<br>
If you want to train intent using new data, insert the data into the 'src/data' folder and run intent_data.py located in the src folder. This will create a unified file named train_intent.csv. Next, run intent_train.py to start learning using the unified data file you just created. Detailed parameter settings can be reconfigured in configs.py to match your data.
<br><br><br>

## 2. Entity Recognition

Second, since you have classified user's intent, we now need to recognize the entities in the conversation. 
In order for us to provide data in conjunction with an external API, we need to pass only the data needed by that API. In this case, BiLSTM-CRF is used to cut only necessary data.
The way to recognize entities using BiLSTM-CRF is based on a [paper by Baidu Research published in 2015](https://arxiv.org/pdf/1508.01991.pdf)
<br><br>
<img src=https://www.depends-on-the-definition.com/wp-content/uploads/2017/11/lstm_crf.png></img>
<br><br>

The parameters used for training are as follows.

        dim = 300
        dim_char = 120
        max_iter = None
        lowercase = True
        train_embeddings = False
        nepochs = 30
        dropout = 0.5
        batch_size = 10
        lr = 0.04
        lr_decay = 0.9
        nepoch_no_imprv = 300
        use_crawler = False
        hidden_size = 512
        char_hidden_size = 128
        crf = True          # size one is not allowed
        chars = True          # if char embedding, training is 3.5x slower
<br>

If you want to learn a new entity, use the existing entity recognizer codes in the entity package, change the data after copying each recognizer. You can also add code to train_entity.py to train your new entity recognizer similar to the existing code. The second parameter indicates whether or not to learn.
<br><br><br>

## 3. API Connection

All the information for the API connection was collected. Finally, we can provide the information in combination with the template senetence which made the information provided by API beforehand.
<br><br>
Here I wrote a code to crawl websites such as Google Translator, Naver Weather, Dust, Sayings, Issues, Restaurants, YouTube Music, and Wikipedia.
<br><br><br>

## 4. Result

#### 4-1. Korean

    AI is awakening now...
    Provided Feature : 날씨, 뉴스, 달력, 맛집, 미세먼지, 명언, 번역, 시간, 위키, 음악, 이슈, 인물
    
    User : 오늘 전주 날씨 알려줘
    A.I : 오늘 전주 날씨를 알려드릴게요. 오늘 전주에는 해가 떴어요. 아주 맑아요. 현재 온도는 25도로 어제보다 0도 높아요

    User : 부산 돼지고기 맛집 추천해줘
    A.I : 부산역 근처에 있는 돼지국밥집 , 본전돼지국밥 에 가보는 건 어떨까요? 
    부산역 5분거리에있는 45년 변함없는 전통의 깊고 깔끔한 돼지고기의 국  물맛으로 유명한 부산의 돼지국밥 전문점입니다. 
    밑반찬으로 제공되는 길게 김치를 가져와 가위로 잘라주는 배추김치와 부추김치가 칼칼한 맛으로 돼지국밥과 잘 어울립니다. 
    돼지국밥은 파가 많이 들어가 있어 시원하고 개운한 맛을 내며 고기가 넉넉히 들어 있어 든든한 한 끼가 가능합니다. 
    부산 현지인이 추천하는 돼지국밥집으로, 평범하지만 진한 국물과 깔끔한 돼지고기가 인상적입니다.  
    운영시간은 매일 08:30  에서  21:30, 주소는 부산 동구 중앙대로214번길 3-8, 전화번호는 051-441-2946,입니다.
    
    User : 사랑해가 영어로 뭐니
    A.I : I love you.
    
    User : 이명박이 누구야
    A.I : 이명박은 대한민국의 제17대 대통령이다. 본관은 경주이다. 
    호는 원래 일송이나 서울특별시장 시절이던 2005년 10월 26일 청계천복원시민위원회 
    전체 회의에 참석한 고대 동문 김정배 위원장의 권유에 따라 청계로 바꿨다. 
    1970년 12월 19일 김윤옥과 결혼하여 1남 3녀를 두었다.
    
    User : 내일 미세먼지 알려줘
    A.I : 어떤 지역을 알려드릴까요?
    User : 송파구
    A.I : 내일 송파구의 미세먼지 정보를 알려드릴게요 , 
    내일 오전 미세먼지 상태는 좋음, 오후 상태는 좋음, 내일 오전 초미세먼지 상태는 좋음, 
    오후 상태는 좋음, 내일 오전 오존 상태는 보통, 오후 상태는 나쁨입니다. 
    공기 상태가 나쁘니 마스크를 착용하세요
    
    User : 지금 몇시니
    A.I : 현재 시각은 11시 15분입니다.
    
    User : 보고싶다가 일본어로 뭐냐
    A.I : 見たいが
    
    User : 명언 들려줘
    A.I :  이 책의 앞표지와 뒤표지는 너무 멀리 떨어져있다.
    
    User : 블랙핑크 노래 들려줘
    A.I : https://www.youtube.com//watch?v=kcSkZRXbops
    
    User : 오늘 며칠이더라
    A.I : 오늘은 2019년 06월 23일입니다.
    
    User : 요즘 이슈 알려줘
    A.I : 오늘의 이슈로는 , 김선우 , 홈플러스 휴무일 , 김병현 , 침향  , 성훈 드라마등 이있습니다.
    
    User : 바보가 무슨뜻이니
    A.I : 어리석고 못나게 구는 사람을 얕잡거나 비난하여 이르는 말 또는 
    지능이 부족하고 어리석어서 정상적으로 판단하지 못하는 사람
    
<br><br>

4-2. English

It currently only works in Korean. The example below is translated to help understand foreigners and does not actually work like this


    AI is awakening now ...
    Provided Feature: Weather, News, Calendar, Restaurant, Fine Dust, Words, Translation, Time, Wiki, Music, Issue, Person
    
    User: Let me know Jeonju weather today
    A.I: Let me know the weather for Jeonju today. Today was the sunrise last week. It is very clear. 
    The current temperature is 25 degrees higher than yesterday.
    
    User: Recommend Pusan Pork Restaurant
    A.I: Why do not you go to the pig rice house near the Busan Station, the main pig?
    It is a specialty of Pusan rice cooked in Busan, which is famous for its deep, 
    clean pork taste of 45 years old tradition, which is 5 minutes away from Busan Station.
    We bring long kimchi which is served as a side dish, and Chinese cabbage kimchi and leek kimchi, 
    which are cut with scissors, match with pork rice soup in a caloric taste.
    Pork cooked rice has a lot of pork, so it has a cool and refreshing taste and it has a lot of meat.
    Pork rice house recommended by Busan locals, but plain but deep soup and refreshing pork are impressive.
    The operating hours are from 08:30 to 21:30 every day, and the address is 
    214, 3-8, Chungang-dong, Dong-gu, Busan, and the phone number is 051-441-2946.
    
    User: I love you in English
    A.I: I love you.
    
    User: Who is Lee Myung Bak
    A.I: Lee is the 17th President of the Republic of Korea. The main building is Gyeongju.
    On October 26, 2005, when Cheonggyecheon Restoration Citizens' Committee
    It changed to Cheonggye in accordance with the recommendation of Kim Jung-bae, 
    an ancient alumni who attended the whole meeting.
    On December 19, 1970, she married Kim Yoon-ok and had three male and one female.
    
    User: Let me know fine dust tomorrow
    A.I: What area do you want me to tell you?
    User: Songpa
    A.I : I'll let you know the fine dust information of Songpa tomorrow,
    Tomorrow morning fine dust condition is good, afternoon condition is good, 
    fine dust condition is good in the early morning of tomorrow,
    The afternoon is good, the morning ozone is normal, and the afternoon is bad.
    Wear a mask because the air condition is poor.
    
    User: I want to see you in Japanese?
    A.I : 見たいが
    
    User: Tell me your good word
    A.I: The front and back cover of this book are too far away.
    
    User: I want to listen Black Pink song
    A.I: https://www.youtube.com//watch?v=kcSkZRXbops
     
    User: Let me know the date today
    A.I: Today is June 23, 2019.
    
    User: Let me know the issues these days
    A.I: Today's issues include Kim Sun-woo, Homeplus holiday, Kim Byung-hyun, Ao-hyang, and Sung-Hoon Drama.
    
    User: What does fool mean?
    A.I: A person who does not know what to say or criticize because he is foolish,
    
<br><br><br>

## 5. Licence

    Copyright 2019 Hyunwoong Go.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
