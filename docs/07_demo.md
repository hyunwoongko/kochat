
## 7. Demo Application

이 챕터에서는 Demo 애플리케이션에 대해 소개합니다. 
데모 애플리케이션은 여행정보를 소개하는 챗봇 애플리케이션으로,
날씨, 미세먼지, 맛집 여행지 정보를 알려주는 기능을 보유하고 있습니다.
Api는 Kochat을 만들면서 함께 만든 [Kocrawl](https://github.com/gusdnd852/kocrawl) 
을 사용했습니다. 
<br><br>

### 7.1. View (HTML)
Html과 CSS를 사용하여 View를 구현하였습니다. 제가 디자인 한 것은 아니고 
[여기](https://bootsnipp.com/snippets/ZlkBn) 에서 제공되는 
부트스트랩 테마를 사용하였습니다.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Kochat 데모</title>

    <script src="{{ url_for('static', filename="js/jquery.js") }}" type="text/javascript"></script>
    <script src="{{ url_for('static', filename="js/bootstrap.js") }}" type="text/javascript"></script>
    <script src="{{ url_for('static', filename="js/main.js") }}" type="text/javascript"></script>

    <link href="{{ url_for('static', filename="css/bootstrap.css") }}" rel="stylesheet" id="bootstrap-css">
    <link href="{{ url_for('static', filename="css/main.css") }}" rel="stylesheet" id="main-css">
    <script>
        greet();
        onClickAsEnter();
    </script>
</head>

<body>
<div class="chat_window">
    <div class="top_menu">
        <div class="buttons">
            <div class="button close_button"></div>
            <div class="button minimize"></div>
            <div class="button maximize"></div>
        </div>
        <div class="title">Kochat 데모</div>
    </div>
    <ul class="messages"></ul>
    <div class="bottom_wrapper clearfix">
        <div class="message_input_wrapper">
            <input class="message_input"
                   onkeyup="return onClickAsEnter(event)"
                   placeholder="내용을 입력하세요."/>
        </div>

        <div class="send_message"
             id="send_message"
             onclick="onSendButtonClicked()">

            <div class="icon"></div>
            <div class="text">보내기</div>
        </div>

    </div>
</div>
<div class="message_template">
    <li class="message">
        <div class="avatar"></div>
        <div class="text_wrapper">
            <div class="text"></div>
        </div>
    </li>
</div>
</body>
</html>
```
<br><br>

### 7.2. 딥러닝 모델 구성
아래와 같은 모델 구성을 사용하였습니다. 

```python
dataset = Dataset(ood=True)
emb = GensimEmbedder(model=embed.FastText())

clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict)
)

rcn = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

kochat = KochatApi(
    dataset=dataset,
    embed_processor=(emb, True),
    intent_classifier=(clf, True),
    entity_recognizer=(rcn, True),
    scenarios=[
        weather, dust, travel, restaurant
    ]
)


@kochat.app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    kochat.app.template_folder = kochat.root_dir + 'templates'
    kochat.app.static_folder = kochat.root_dir + 'static'
    kochat.app.run(port=8080, host='0.0.0.0')
```
<br><br>

### 7.3. 시나리오 구성
Kocrawl을 이용해 4가지 의도에 맞는 시나리오를 구성하였습니다.
```python
weather = Scenario(
    intent='weather',
    api=WeatherCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

dust = Scenario(
    intent='dust',
    api=DustCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

restaurant = Scenario(
    intent='restaurant',
    api=RestaurantCrawler().request,
    scenario={
        'LOCATION': [],
        'RESTAURANT': ['유명한']
    }
)

travel = Scenario(
    intent='travel',
    api=MapCrawler().request,
    scenario={
        'LOCATION': [],
        'PLACE': ['관광지']
    }
)
```
<br><br>

### 7.4. Javascript 구현 (+ Ajax)
마지막으로 버튼을 누르면 메시지가 띄워지는 애니메이션과 Ajax를 통해
Kochat 서버와 통신하는 소스코드를 작성하였습니다. 간단한 chit chat 대화
3가지 (안녕, 고마워, 없어)는 규칙기반으로 구현하였습니다. 추후에
Seq2Seq 기능을 추가하여 이 부분도 머신러닝 기반으로 변경할 예정입니다.

```javascript
// variables
let userName = null;
let state = 'SUCCESS';

// functions
function Message(arg) {
    this.text = arg.text;
    this.message_side = arg.message_side;

    this.draw = function (_this) {
        return function () {
            let $message;
            $message = $($('.message_template').clone().html());
            $message.addClass(_this.message_side).find('.text').html(_this.text);
            $('.messages').append($message);

            return setTimeout(function () {
                return $message.addClass('appeared');
            }, 0);
        };
    }(this);
    return this;
}

function getMessageText() {
    let $message_input;
    $message_input = $('.message_input');
    return $message_input.val();
}

function sendMessage(text, message_side) {
    let $messages, message;
    $('.message_input').val('');
    $messages = $('.messages');
    message = new Message({
        text: text,
        message_side: message_side
    });
    message.draw();
    $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);
}

function greet() {
    setTimeout(function () {
        return sendMessage("Kochat 데모에 오신걸 환영합니다.", 'left');
    }, 1000);

    setTimeout(function () {
        return sendMessage("사용할 닉네임을 알려주세요.", 'left');
    }, 2000);
}

function onClickAsEnter(e) {
    if (e.keyCode === 13) {
        onSendButtonClicked()
    }
}

function setUserName(username) {

    if (username != null && username.replace(" ", "" !== "")) {
        setTimeout(function () {
            return sendMessage("반갑습니다." + username + "님. 닉네임이 설정되었습니다.", 'left');
        }, 1000);
        setTimeout(function () {
            return sendMessage("저는 각종 여행 정보를 알려주는 여행봇입니다.", 'left');
        }, 2000);
        setTimeout(function () {
            return sendMessage("날씨, 미세먼지, 여행지, 맛집 정보에 대해 무엇이든 물어보세요!", 'left');
        }, 3000);

        return username;

    } else {
        setTimeout(function () {
            return sendMessage("올바른 닉네임을 이용해주세요.", 'left');
        }, 1000);

        return null;
    }
}

function requestChat(messageText, url_pattern) {
    $.ajax({
        url: "http://0.0.0.0:8080/" + url_pattern + '/' + userName + '/' + messageText,
        type: "GET",
        dataType: "json",
        success: function (data) {
            state = data['state'];

            if (state === 'SUCCESS') {
                return sendMessage(data['answer'], 'left');
            } else if (state === 'REQUIRE_LOCATION') {
                return sendMessage('어느 지역을 알려드릴까요?', 'left');
            } else {
                return sendMessage('죄송합니다. 무슨말인지 잘 모르겠어요.', 'left');
            }
        },

        error: function (request, status, error) {
            console.log(error);

            return sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'left');
        }
    });
}

function onSendButtonClicked() {
    let messageText = getMessageText();
    sendMessage(messageText, 'right');

    if (userName == null) {
        userName = setUserName(messageText);

    } else {
        if (messageText.includes('안녕')) {
            setTimeout(function () {
                return sendMessage("안녕하세요. 저는 Kochat 여행봇입니다.", 'left');
            }, 1000);
        } else if (messageText.includes('고마워')) {
            setTimeout(function () {
                return sendMessage("천만에요. 더 물어보실 건 없나요?", 'left');
            }, 1000);
        } else if (messageText.includes('없어')) {
            setTimeout(function () {
                return sendMessage("그렇군요. 알겠습니다!", 'left');
            }, 1000);


        } else if (state.includes('REQUIRE')) {
            return requestChat(messageText, 'fill_slot');
        } else {
            return requestChat(messageText, 'request_chat');
        }
    }
}
```
<br><br>

### 7.5. 실행 결과

![](https://user-images.githubusercontent.com/38183241/86410173-4347a680-bcf5-11ea-9261-e272ad21ed36.gif)
<br><br>

#### ⚠ Warning

데모 데이터셋은 양이 적기 때문에 다양한 지명이나 다양한
음식, 다양한 여행지 등은 알아 듣지 못합니다. (데모영상을 위해
일부 서울 지역 위주로만 데이터셋을 작성했습니다.) 데모데이터셋은
데모영상을 찍기 위한 아주 작은 dev 데이터 셋입니다.
실제로 다양한 도시나 다양한 음식 등을 알아 들을 정도로 대화를 나누려면 데모 데이터셋보다 
자체적인 데이터 셋을 많이 삽입하셔야 더욱 좋은 성능을 기대할 수 있을 것입니다. 
하루빨리 Pretrain 모델을 지원하여 이러한 문제를 해결하도록 하겠습니다. 
모든 데모 애플리케이션 소스코드는 
[여기](https://github.com/gusdnd852/kochat/tree/master/demo) 를 참고해주세요
