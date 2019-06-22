# Chatbot


This repository introduces a simple chatbot architecture based on Deep Learning. The architecture includes intent classification of conversations, entity recognition, and providing information through various API connection.
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
  * GPU : Ndivia RTX 2070
<br><br><br>

## 1. Intent Classification

First, if the user tries to talk, he or she must know the intent of the conversation.
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

## 4. Licence

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
