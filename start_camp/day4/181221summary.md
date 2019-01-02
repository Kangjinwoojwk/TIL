# 181221 학습일지

## 1.챗봇

* 텔레그램으로 할 것, 카카오톡 챗봇은 12/3으로 마감, 챗봇 많이 쓰니까 드래그 앤 드롭으로 쉽게 하게 하고 돈을 받자! 그래서 막혀...이런....

* c9의 flask, url추가 계속 가능 계속 돌아가게 하려면 export FLASK_ENV='development'

* flask run -h 0.0.0.0 -p 8080     flask 가동

  ```python
  #-*- coding:utf-8 -*-     #맨 위에 쓰면 한글 문제 사라짐
  
  if __name__=='__main__':  #마지막에 있어야 한다. 이거 있는 순간 밑은 다 무시
      app.run(host='0.0.0.0', port=8080)  
  ```
  * python3 app.py  만으로 flask를 가동하게 하는 코드

* @app.route('/ide/\<string:username>/\<string:workspace>')  #어떤 값인지 모르지만 string으로 인식해서 username 과 workspace로 쓰겠다

* 라우트를 정해두면 그에 따라 원하는거 리턴가능, 입력값을 받을 수도 있다.

* 사이트 연결, templates 폴더를 만든다. 약속이다

  flask내에 html의 {{}}는 플라스크에서만 제공하는 기능, 해당 기능을 통해 변수를 넘겨 줄 수 있다.

  ```python
  app = Flask(__name__)
  @app.route('/ide/<string:username>/<string:workspace>')
  def username_workspace(username, workspace):
      return render_template('ide.html',username=username, workspace=workspace)
  
  @app.route('/')
  def index():
      lunch=random.choice(['20층', 'Diet'])
      return render_template('index.html', lunch=lunch)
  ```

  index.html, lunch를 받고 있다.

  ```html
  <h1>Hi</h1>  
  <h2>Lunch: {{ lunch }}</h2>
  ```

  ide.html, username과 workspace를 받고 있다

  ```html
  <h1>{{ username }}</h1>
  <p>
      {{ workspace }}
  </p>
  ```

* sudo pip3 install beautifulsoup4  beatifulsoup 깔기-파싱하기 위함

* 입력창을 만들어 보자

* 라우트 설정-길 뚫어 주는 것, 요청이 들어 올 수 있는 길, 길을 뚫었다. 뭘 해야 하나?

* 요청-응답사이클

!tab하면  html은 기본은 나온다, input 은 셀프 클로징의 대표적, 뭔가 깔았으면 서버 껏다 켜야 된다.

변수만 있는 상황, 이름을 지어줘야 한다. 이름이 있어야 변수를 잡을 수 있다. 이름이 있어야 돼

* 핑, 퐁방식, 올때 url 로와, 이름으로 받아야 돼

## 2. 챗봇

* https://api.telegram.org/bot710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0
* 내 챗봇, 봇에대한 내용나와
* https://api.telegram.org/bot710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0/getUpdates
* 업데이트 받는 곳
* 내 id 718649676, from에 있다
* https://api.telegram.org/bot710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0/sendMessage?chat_id=718649676&text=HappyHacking
* 치면 나한테 HappyHacking이 온다 요청 하는 것
* 모두 대문자로 쓰면 보통은 상수라고 생각한다-컨벤션
* c9에선 telegram에 못들어 간다.
* 그래서 우회 필요, https://api.hphk.io/telegram/bot{}/sendMessage?chat_id={}&text={}'.format(BOT_TOKEN,MY_CHAT_ID,msg)
* 무슨 일을 할 수 있는가는 달라, 공장에 불량률 높? 낮? 포장이 제대로 안돼? 높은 데는 엑스레이? 낮은데 가보니 선풍기

```python
#-*- coding:utf-8 -*-
from flask import Flask, jsonify, render_template, request
import random
import lotto_functions
import requests

app = Flask(__name__)
@app.route('/ide/<string:username>/<string:workspace>')
def username_workspace(username, workspace):
    return render_template('ide.html',username=username, workspace=workspace)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/search')
# def search():
    

@app.route("/ping")
def ping():
    return render_template('ping.html')

@app.route("/pong")
def pong():
    ssum = request.args.get('ssum')
    me = request.args.get('me')
    match_point = random.choice(range(1,100))
    result = me + '=>' + ssum
    MY_CHAT_ID = '718649676'
    BOT_TOKEN = '710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0'
    url = 'https://api.hphk.io/telegram/bot{}/sendMessage?chat_id={}&text={}'.format(BOT_TOKEN,MY_CHAT_ID,result)
    response = requests.get(url)
    return render_template('pong.html', ssum = ssum, match_point = match_point, me = me)
    
@app.route("/throw")
def throw():
    return render_template('throw.html')







    
@app.route("/google")
def google():
    Search = request.args.get('Search')
    result = '한 사용자가 구글에서 '+Search+'을/를 검색하였습니다.'
    MY_CHAT_ID = '718649676'
    BOT_TOKEN = '710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0'
    url = 'https://api.hphk.io/telegram/bot{}/sendMessage?chat_id={}&text={}'.format(BOT_TOKEN,MY_CHAT_ID,result)
    response = requests.get(url)
    return render_template('google.html', Search=Search)
@app.route("/naver")
def naver():
    Search = request.args.get('Search')
    result = '한 사용자가 네이버에서 '+Search+'을/를 검색하였습니다.'
    MY_CHAT_ID = '718649676'
    BOT_TOKEN = '710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0'
    url = 'https://api.hphk.io/telegram/bot{}/sendMessage?chat_id={}&text={}'.format(BOT_TOKEN,MY_CHAT_ID,result)
    response = requests.get(url)
    return render_template('naver.html', Search=Search)
@app.route("/daum")
def daum():
    Search = request.args.get('Search')
    result = '한 사용자가 다음에서 '+Search+'을/를 검색하였습니다.'
    MY_CHAT_ID = '718649676'
    BOT_TOKEN = '710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0'
    url = 'https://api.hphk.io/telegram/bot{}/sendMessage?chat_id={}&text={}'.format(BOT_TOKEN,MY_CHAT_ID,result)
    response = requests.get(url)
    return render_template('daum.html', Search=Search)
@app.route("/baidu")
def baidu():
    Search = request.args.get('Search')
    result = '한 사용자가 바이두에서 '+Search+'을/를 검색하였습니다.'
    MY_CHAT_ID = '718649676'
    BOT_TOKEN = '710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0'
    url = 'https://api.hphk.io/telegram/bot{}/sendMessage?chat_id={}&text={}'.format(BOT_TOKEN,MY_CHAT_ID,result)
    response = requests.get(url)
    return render_template('baidu.html', Search=Search)

@app.route('/get_lotto/')
def get_lotto():
    turn = request.args.get('turn')
    result = '한 사용자가 로또 '+turn+'회차를 검색하였습니다.'
    MY_CHAT_ID = '718649676'
    BOT_TOKEN = '710115390:AAE59rdYXSUJY2pXJJr4i01JBShF8hMXRF0'
    url = 'https://api.hphk.io/telegram/bot{}/sendMessage?chat_id={}&text={}'.format(BOT_TOKEN,MY_CHAT_ID,result)
    response = requests.get(url)
    data = lotto_functions.get_lotto(request.args.get('turn'))
    return jsonify(data)

@app.route('/hi')
def hi():
    return 'Hello ssafy'
    
@app.route('/pick_lotto')
def pick_lotto():
    return random.sample(range(1,46),6)
    


if __name__=='__main__':
    app.run(degug=True, host='0.0.0.0', port=8080)
```

기본 봇 제작 코드