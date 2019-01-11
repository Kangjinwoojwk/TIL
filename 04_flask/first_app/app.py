from flask import Flask, jsonify #jsonify는 리스트를 세상 밖으로 보내 주는 것
import random

app = Flask(__name__)

@app.route('/')# 루트, 모든 것의 최상단, 라우팅을 한다
def index():
    return 'Hi'

@app.route('/ssafy')
def ssafy():
    return 'sssssssafy'

@app.route('/pick_lotto')
def pick_lotto():
    pick_numbers = random.sample(range(1,46),6)
    pick_numbers.sort()
    #return str(pick_numbers)
    return jsonify(pick_numbers)

@app.route('/hi/<string:name>')#변수를 name으로 받을거다.variable 라우팅
def hi(name): #name을 함수 내에서 쓸 수 있게 넣어준다. 함수 정의 중요
    return (f'hi {name}!')

@app.route('/dictionary/<string:word>')
def dictionary(word):
    en = []
    kr = []
    result_f = open('dictionary.txt','rt',encoding='UTF8')
    for line in result_f:
        (e,k) = line.split('\t')
        en.append(e)
        kr.append(k[:-1])
    result_f.close()
    my_dictionary = dict(zip(en, kr))
    if my_dictionary.get(word):
        return (f'{word}은(는) {my_dictionary[word]}!')
    else:
        return (f'{word}은(는) 나만의 단어장에 없는 단어입니다!')

# if __name__ == '__main__':
#     app.run(debug = True)
