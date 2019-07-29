# Tensorflow

## 설치

* 우선 텐서 플로우 부터 설치하자

```
pip3 install --upgrade tensorflow
```

* 그래픽 카드를 사용하고 싶다면 http://docs.nvidia.com/cuda 에서 cuda를 설치하자

```
pip3 install --upgrade tensorflow-gpu
```

* 라이브러리 설치, 수치계산, 그래프 출력, 이미지처리

```
pip3 install numpy matplotlib pillow
```

* 주피터 노트북 설치

```
pip3 install jupyter
```

## 텐서플로 프로그래밍 101

```python
import tensorflow as tf #텐서플로우 추가

hello = tf.constant('Hello, TensorFlow!') # 선언
print(hello) # 출력
```

> ```
> Tensor("Const:0", shape=(), dtype=string)
> ```

### 랭크와 셰이프

```
3 # 랭크가 0인 텐서; 셰이프는 []
[1. ,2. ,3.] # 랭크가 1인 텐서; 셰이프는 [3]
[[1., 2., 3.], [4., 5., 6.]] # 랭크가 2인 텐서; 셰이프는 [2, 3]
[[[1., 2., 3.]][[7., 8., 9.]]] # 랭크가 3인 텐서; 셰이프는 [2, 1, 3]
```

* 랭크: 차원수, 0이면 스칼라, 1이면 벡터, 2면 행렬, 3이상이면 n차원 텐서라고 한다.

* 셰이프: 각차원의 요소 개수, 텐서 구조 설명
* dtype은 요소들의 자료형, string, float, int 등

```python
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)
print(c)
```

> ```
> Tensor("Add:0", shape=(), dtype=int32)
> ```

왜 42가 아닌가? 텐서플로는 구조가 그래프 생성과 그래프 실행으로 분리 되어 있다.

위는 생성만 할 뿐

```python
sess = tf.Session()
print(sess.run(hello))
print(sess.run([a,b,c]))
```

> ```
> b'Hello, TensorFlow!'
> [10, 32, 42]
> ```

실제로 연산이 일어 나는 곳은 이곳

* 그래프: 텐서 연산 모음

텐서 플로는 텐서의 연산들을 먼저 정의하여 그래프를 만들고 이후 필요할 때 연산을 실행하는 코드 넣어 원하는 시점에 연산, `지연실행`, 이 방식으로 실제 계산은 C++로 구현한 코어 라이브러리에서 하기에 성능이 향상 된다.마지막에 close()해줄것

### 플레이스홀더와 변수

* 플레이스 홀더: 그래프에 사용할 입력값을 나중에 받기 위해 사용하는 매개변수
* 변수: 그래프를 최적화하는 용도로 텐서플로가 학습한 결과를 갱신하기 위해 사용하는 변수, 신경망 성능 좌우, 일종의 메타 프로그래밍

```python
X = tf.placeholder(tf.float32, [None, 3]) # None은 크기가 아직 정해지지 않았다는 것
print(X)
x_data = [[1,2,3], [4, 5, 6]]
W = tf.Variable(tf.random_normal([3,2])) # [3,2]행렬형 정규분포 무작위값 같는 텐서
b = tf.Variable(tf.random_normal([2,1])) # []
expr = tf.matmul(X, W) + b
```

```python
sess = tf.Session() # 실행부 오픈
sess.run(tf.global_variables_initializer()) # 변수 초기화, 기존에 학습한 값이 있어서 가져와서 쓸 것 아니면 꼭 초기화 해야 한다.
print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X:x_data}))

sess.close()
```

### 선형 회귀 모델 구현

* 선형 회귀: 주어진 x,y 로 서로간의 관계 파악, x주어졌을때 y파악

플레이스 홀더에 이름을 부여하지 않으면 Placeholder_1식으로 자동부여, 이름 부여하는 편이 어떠한 텐서가 어떻게 사용 되는지 알 수 있다. 텐서보드에서도 이름 출력, 디버깅 수월, 변수와 연산 또는 연산 함수도 이름 지정 가능

* 손실 함수: 한싼의 데이터에 대한 손실값을 계산하는 함수, 실제 값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타내는 값, 작을수록 제대로 설명
* 비용: 전체 데이터에 대한 손실값의 합, 손실값은 예측값과 실제값의 거리를 가장 많이 사용
* 학습: 변수 값을 다양하게 넣어 보면서 손실값 최소화하는 W, b 구하는 것

* 경사 하강법:최적화 기본 알고리즘, 그래프와 같이 함수의 기울기를 구하고 기울기가 낮은 쪽으로 계쏙 이동시키면서 최적의 값을 찾아 나가는 방법

* 학습률(learning_rate):학습을 얼마나 급하게 할 것인가 설정하는 값, 너무 크면 최적 못찾고 지나치고 작으면 학습 속도 느려
* 하이퍼파라미터:학습진행 과정에 영향을 주는 변수ex)학습률, 잘 튜닝하는게 큰 과제

```python
import tensorflow as tf
x_data = [1,2,3]
y_data = [1, 2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # W,b 각각 -1.0~1.0 값의 균등분포 무작위값으로 초기화
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
X = tf.placeholder(tf.float32, name="X") # 자료입력 받을 플레이스 홀더 설정
Y = tf.placeholder(tf.float32, name="Y")
hypothesis = W*X+b # X,Y 상관관계를 분석하기 위한 수식 작성, W와의 곱, b와의 합으로 설명, W는 가중치 b는 편향
# W,x가 행렬 아니므로 기본곱셈 연산자 사용
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1) # 경사하강법을 이용한 최적화 함수를 이용 손실값 최소 연산 그래프 생성
train_op = optimizer.minimize(cost)
with tf.Session() as sess: # with가 끝나면 알아서 close된다.
    sess.run(tf.global_variables_initializer()) # 초기화
    for step in range(100): # 100번에 걸쳐서 학습시킨다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y:y_data}) # 학습에서 받는 값을 할당한다.
        print(step, cost_val, sess.run(W), sess.run(b))
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X:5})) # 학습에 없는 값을 넣을때의 Y를 알아본다
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X:2.5}))
```

100번의 학습을 하느 엇비슷하게 나온다.

## 기본 신경망 구현

심층, 다층 신경망 간단 구현

### 인공신경망의 작동원리

* 인공 신경망: 뉴런에 기초, 뉴런의 전달 신호 세기가 더 강하게 되기도 약하게 되기도한다. 가중치, 편향 처리후 활성화 함수 거쳐 결과 y를 만든다.
* 학습(훈련):변경하면서 적절한 값을 찾아내는 최적화 과정
* 활성화 함수: 인공 신경망 통한 값, 최종 전달값 만듬, ex)sigmoid, ReLU, tanh

인공뉴런은 가중치에 활성화 함수를 연결한 간단한 값이지만 이런 간단한 개념을 충분히 많이 이어 놓은 것만으로 인간이 인지하기 어려운 매우 복잡한 패턴도 스스로 학습 가능하다.

수천~수만개 조합 일일이 변경하며 계산 오래 걸려서 훈련 불가, 층 깊어질수록 기하 급수적 증가로 더 불가

* 제한된 볼트만 머신:제프리 힌튼 교수가 개발한 신경망 학습 알고리즘, 심층 신경말 효율적 학습 증명

이후 드롭아웃 기법, ReLU 등 활성 함수 개발되며 발전, 단순 수치계산과 병렬처리 능한 GPU 발전 덕에 딥러닝 시대 개막

* 역전파: 출력층이 내놓은 결과의 오차를 신경망을 따라 입력층까지 역으로 전파하여 계산하는 방식, 입력층부터 가중치를 조절해가는 기존 방식보다 유의미, 가중치를 조절해서 최적화 과정이 빠르고 정확

다층신경망 학습 가능하게 한 역전파는 1985년 루멜하트가 제안, 데이터와 연산 문제로 파묻혀 있다가 연구 계속되면서 재조명된 알고리즘

역전파는 신경망 구현에 꼭 필요하지만 구현은 어려워, 텐서플로는 활성화 함수와 학습함수를 기본적으로 역전파 기법 제공, 드롭아웃이나 ReLU같은 활성함수도 직접 구현할 필요는 없다. 역전파는 매우 중요, 추후 딥러닝 연구하고자 한다면 다시 들려다 볼 것

### 간단한 분류모델 구현

딥러닝, 가장 폭넓게 사용되는 분야는 패턴 인식 영상처리

* 분류: 패턴을 파악해 여러 종류로 구분하는 작업
* 원-핫 인코딩:데이터가 가질 수 있는 값들을 일렬로 나열한 배열을 만들고 그중 표현하려는 값을 뜻하는 인덱스의 원소만 1로 표기하고 나머지는 0으로 채우는 표기법
* 교차 엔트로피:예측값과 실제값 사이의 확률 분포차이를 계산한 값

```python
# 기본 함수
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model),axis=1))
```

* 손실함수(비용함수), cost는 여기에서 따왔습니다.

reduce_xxx 함수는 텐서의 차원 줄여줌, xxx는 축소 방법, axis매개별수로 축소할 차원 정해줌, reduce_sum(<입력 텐서>,axis=1)은 주어진 텐서의 1번째 차원의 값들을 다 더해 값 1개로 만들어서 그 차원을 없앤다는 뜻, sum외에 prod, min, max, mean, all(논리적 AND), any(논리적 OR), logsumexp등을 제공

```python
import tensorflow as tf # 텐서플로우 임포트
import numpy as np # 수치해석용 넘파이 임포트
# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]) # 학습에 사용할 데이터 정의 털과 날개가 있느냐,있으면 1, 없으면 0
y_data = np.array([
    [1, 0, 0], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])
X = tf.placeholder(tf.float32) # 실측 값 넣을 것이니 X,Y는 플레이스 홀더로 설정
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.)) # 가중치 W는 입력층, 출력층의 구선인 2, 3으로 설정
b = tf.Variable(tf.zeros([3])) # 편향 변수 b는 레이블 수인 3개 요소 가진 변수로 설정
L = tf.add(tf.matmul(X,W),b) # 가중치 곱하고 편향치 더함
L = tf.nn.relu(L) # 활성화 함수 ReLu 연결
model = tf.nn.softmax(L) # softmax 함수는 결괏값의 전체 합이 1이 되도록 만드렁 준다. 전체가 1이니 각각은 해당 결과의 확률로 해석 가능
# 교차 엔트로피 기본함수
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model),axis=1))
# 경사하강법으로 최적화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로의 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행
for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    
    # 학습 도중 10번에 한번씩 손실값 출력
    if step%10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실제값:', sess.run(target, feed_dict={Y:y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
```

정확도 66%에서 많이 잘 안바뀐다. 신경망이 한층이기 때문, 늘리면 쉽게 해결된다.

### 심층 신경망 구현

* 딥러닝: 둘 이상의 신경망을 구성하는 심층 신경망, 기본은 간단. 앞서 만든 신경망에 가중치와 편향 추가

입력층과 출력층은 각각 특징과 분류 개수로 맞추고, 중간의 연결 부분은 맞닿은 층의 뉴런 수와 같도록 맞추면 된다. 중간 연결 부분을 은닉층이라 하며 은닉층의 뉴런 수는 하이퍼파라미터, 실험을 통해 가장 적절한 수를 정하면 된다.

단층에서는 출력층에 활성화 함수 사용, 일반적이지 않아, 하이퍼 파라미터와 같이 은닉, 출력층에서 활성화 함수 적용할지 정하는 일 또한 신경함 만드는게 가장 중요한 경험적, 실험적 요소

```python
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.)) # 가중치 2는특징 10은 은닉층 뉴런수
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.)) # 10은 은닉층 뉴런 수, 3은 분류수

b1 = tf.Variable(tf.zeros([10])) # 은닉층 뉴런수
b2 = tf.Variable(tf.zeros([3])) # 분류수

L1 = tf.add(tf.matmul(X, W1), b1) # 식적용
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행
for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    
    # 학습 도중 10번에 한번씩 손실값 출력
    if step%10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실제값:', sess.run(target, feed_dict={Y:y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
```

## 텐서보드와 모델 재사용

학습시킨 모델을 저장하고 재사용하는 방법, 텐서보드를 이용해 손실값의 변화를 그래프로 추적하는 법

### 학습 모델 저장하고 재사용

* 털, 날개, 기타, 포유류, 조류 넣은 원핫 인코딩

신경망 계층 수와 은닉층의 뉴런수를 늘리면 복잡도가 높은 문제 해결 하는데 도움이 된다. 그러나 이런다고 무조건 도움 되는 것은 아니고 오히려 과적합이라는 문제에 빠질 수 있다. 계층과 뉴런 수를 최적화하는 것이 중요하다.

```python
import tensorflow as tf
import numpy as np

# np로 데이터 읽어 들이고 변환
data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# 모델 저장 할때 쓸 변수 생성, 학습 사용x, 학습 횟수 카운트, trainable= False
global_step = tf.Variable(0, trainable = False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망을 하나 더 늘리고 대신 가중치만 넣고 편향을 빼보자 구성이 명확해진다.
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
# global_step을 넘겨 줬다. 이렇게 하면 최적화 함수가 변수 최적화 할때마다 변수값 1씩 증가
train_op = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables()) # global_variables()는 앞서 정의한 변수들을 가져오는 함수

# ./model 디렉토리에 기존 학습 모델 있는지 확인해서 있다면 saver.restore로 가져오고 아니면 초기화
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
    
# 가져온 것이니 2번만 재학습
for step in range(2):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X:x_data, Y:y_data}))
          
# 지정한 파일에 저장
saver.save(sess, './model/dnn.ckpt', global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y:y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
```

한 스텝 할때마다 학습한다, step이 global_step이라는 변수로 되어 있어서 그거만 따로 증가한다.

이 방식을 이용하여 모델 구성, 학습, 예측을 각각 분리하여 학습한 뒤 예측만 단독으로 실행하는 프로그램을 작성할 수 있습니다.

### 텐서보드 사용

* 텐서보드: 다른 라이브러리와 프레임 워크 두고 왜 텐서플로인가? 텐서 보드의 역할이 크다! 학습 시간 오래 걸린다. 모델을 효과적으로 실험하려면 학습 과정 추적중요, 번거로움 추가 작업 들어가, 텐서보드는 이걸 해결해주는 도구

```python
import tensorflow as tf
import numpy as np

# 텐서보드를 이용하기 위한 코딩을 해보자. 우선 앞에것 그대로 읽는다.
data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(0, trainable = False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 신경망 계층마다 코드 덧붙이기
# with tf.name_scope로 묶은 블록은 텐서보드에서 한 계층 내부를 표현
# 이름 붙이면 해당 이름의 변수가 어디에 쓰이는지 쉽게 확인가능
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

    # tf.summary.scalar 를 이용해 수집하고 싶은 값들을 지정할 수 있습니다.
    tf.summary.scalar('cost', cost) # 손실값 추정을 위해 수집할 값 지정 코드, cost 텐서값 손쉽게 지정가능
    tf.summary.histogram('Weights', W1) # 가중치, 편향 등의 변화를 그래프로 보고 싶다면 추가
    
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    print('Step: %d, ' % sess.run(global_step),
         'Cost: %.3f' % sess.run(cost, feed_dict={X:x_data, Y: y_data}))
    summary = sess.run(merged, feed_dict={X: x_data, Y:y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))
saver.save(sess, './model/dnn.ckpt', global_step=global_step)
prediction = tf.argmax(model, 1)
target = tf.argmax(Y,1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실제값:', sess.run(target, feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

log폴더에 기록된다. 

`tensorboard --logdir=./logs`으로 서버를 키고

`[http://localhost:6006](http://localhost:6006/)`으로 들어가면 그래프를 볼 수 있다.

#### 더보기

* 텐서플로서빙: 학습 모델을 실제 서비스에 적용하기 쉽게 만들어 주는 서버 환경, 학습된 모델을 사용하는 프로그램을 텐서플로로 직접 만들 수도 있지만 서빙은 쉬운 모델 변경, 여러 모델 한 서버 서비스 등 편의 기능 제공

https://tensorflow.github.io/serving/

## 헬로 딥러닝 MNIST

* MNIST: 손으로 쓴 숫자들의 이미지를 모아놓은 데이터셋, 0~9까지 28x28픽셀

http://yann.lecun.com/exdb/mnist

### MNIST 학습

* 정리는 잘되어 있지만 사용위해선 내려 받고 읽고 나누고 학습에 적합한 형식으로 처리하는 과정 거쳐야돼, 텐서 플로 이용하면 쉽고 간편

`gunzip [file name]` gz 파일 압축해제

* 미니배치: 제반 여건 때문에 데이터를 적당한 크기로 잘라서 학습 시키는 것, 이미지 하나씩 학습보다 여러 개 한꺼번에 학습하는게 효과가 좋지만 메모리, 성능 필요

왜 테스터 데이터 따로? 학습 잘됐는지 알기 위함, 학습 데이터에 과적합되는 것을 막기 위함

* 에포크:학습데이터 전체 한바퀴 도는 것

### 드롭아웃

* 드롭아웃: 과적합을 막기 위한 방법 중 하나, 효과 좋다. 원리 간단, 학습시 전체 신경망 중 일부만을 사용하도록 하는 것, 학습 단계마다 일부 뉴런 제거로 일부 특징이 고정 되는 것을 막아 가중치에 균형을 잡게 한다. 일부씩 학습하기에 시간은 오래 걸린다.

* 과적합: 학습 결과가 학습데이터에 매우 잘 맞아서 그 외의 데이터에 잘 안맞는 상황

dropout으로 학습시에도 예측시에는 신경망 전체를 활용 하도록 해야 한다.

* 배치 정규화: 최근 드롭 아웃 대신 많이 쓰이는 것, 학습 속도 향상의 장점도 있다. 원래는 과적합 아닌 발산, 소실 막기 위한 것, 밑의 코드로 간단히 사용가능

```python
tf.layers.batch_nomalization(L1, training = is_traing)
```

### matplotlib

* 시각화 그래프 쉽게 그려주는 파이썬 라이브러리, 학습 결과를 손글씨 이미지로 확인하는 예제 만들자

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)#데이터 내려 받고 레이블을 원핫인코딩방식으로 읽는다.
X = tf.placeholder(tf.float32, [None,784]) # None은 크기가 정해지지 않았다는 것, 한번에 학습시킬 MNIST 이미지 개수 지정 값
Y = tf.placeholder(tf.float32, [None,10]) # 배치 크기 미리 정해도 되지만 학습 개수 계속 바꿔가면서 실험하려는 경우 None넣으면 알아서 계산
# 신경망 구축
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X,W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)
# 손실값 구하고 평균 송실값 구하기, 최적화 수행하도록 그래프 구성
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost],
                              feed_dict={X:batch_xs, Y:batch_ys})
        total_cost += cost_val
    print('Epoch:','%04d' % (epoch+1),
         'Avg.cost = ', '{:3f}'.format(total_cost / total_batch))
print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:',sess.run(accuracy,
                     feed_dict={X:mnist.test.images,
                               Y:mnist.test.labels}))
keep_prob = tf.placeholder(tf.float32) # 학습에는 떼놔도 예측시에는 전부 사용 하도록 설정하기 위한 것

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob) # dropout 쓰기만 하면 작용, 0.8은 80% 뉴런을 사용하겠다는 것

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1,W2))
L2 = tf.nn.dropout(L2, keep_prob)

_, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y: batch_ys, keep_prob:0.8}) # 0.8은 80%의 뉴런만 쓰겠다는 것
print('정확도:', sess.run(accuracy, feed_dict={X:mnist.test.images,
                                           Y: mnist.test.labels,
                                           keep_prob:1})) # 예상할때는 전부 사용
keep_prob = tf.placeholder(tf.float32) # 학습에는 떼놔도 예측시에는 전부 사용 하도록 설정하기 위한 것

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob) # dropout 쓰기만 하면 작용, 0.8은 80% 뉴런을 사용하겠다는 것

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1,W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256,10], stddev=0.01))
model = tf.matmul(L2,W3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for eposh in range(3):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run([optimizer, cost],
                              feed_dict = {X:batch_xs,
                                          Y:batch_ys,
                                          keep_prob:0.8})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch+1),
         'Avg. cost =', '{:.3f}'.format(total_cost/total_batch))
print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))

import matplotlib.pyplot as plt
import numpy as np

labels = sess.run(model, feed_dict={X: mnist.test.images,
                                   Y: mnist.test.labels,
                                   keep_prob: 1})
fig = plt.figure()
for i in range(10): # 2행 5열 그래프 제작, i+1번째 숫자에 이미지 출력
    subplot = fig.add_subplot(2, 5, i + 1) # 깨끗한 이미지 위해 x,y 눈금 미출력
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i])) # 출력 이미지 위 예측 숫자 출력, np.argmax는 tf.argmax와 같은 기능
    #결괏값인 labels의 i번째 요소가 원-핫 인코딩 형식으로 되어 있으므로, 해당 배열에가 가장 높은 값을 가진 인덱스를 예측한 숫자로 출력
    # 1차원 배열로 되어 있는 i번째 이미지 데이터를 28x28 형식의 2차원 배열로 변형하여 이미지 형태로 출력
    # cmap 파라미터를 통해 이미지를 그레이 스케일로 출력
    subplot.imshow(mnist.test.images[i].reshape((28,28)), cmap=plt.cm.gray_r)
    
plt.show()
```

## CNN

### CNN

신경망 구성 다양, 방식 따라 해결 성능 달라지고 새로운 방식 사용가능, 신경망 구성 방식 연구가 신경망 학습에 중요

* CNN(합성곱 신경망): 얀 레쿤 교수가 1998년 제안, 강력한 성능, 음성인식이나 자연어 처리에도 응용 중

### CNN 개념

* 컨볼루션 계층: CNN에서 가중치와 편향을 적용하는 계층
* 풀링 계층: 컨볼루션에서 계산된 값들 중 하나의 선택해서 가져 오는 계층

개념은 간단, 2차원의 평면 행렬에서 지정한 영역의 값들을 하나의 값으로 압축, 컴볼루션에서 가중치와 편향 적용, 풀링은 선택

일정 크기의 윈도우를 설정, 오른쪽, 아래쪽으로 한칸씩 움직이며 은닉층 제작, 움직이는 크기 변경 가능

* 스트라이드: CNN에서 몇 칸씩 움직일지 정하는 값

윈도우 하나를 은닉층의 뉴런 하나로 압축 할때, 컴볼루션 계층에서 윈도우 크기만큼의 가중치와 1개의 편향을 적용, 윈도우 크기가 3x3이면 3x3개의 가중치와 1개의 편향

* 커널(필터):윈도우가 은닉층 뉴런 하나로 압축 될때 필요한 가중치와 편향, 은닉층을 만들기 위한 모든 윈도우에 공통 적용

가중치를 만드는 것이 CNN의 특징, 원래라면 28x28 입력층이 있으면 기본 신경망으로 모든 뉴런 연결하면 784개의 가중치를 찾아야 하지만, 컨볼루션 계층에서는 3x3인 커널만 찾으면 된다. 계산량이 줄어서 학습이 더 빠르고 효율적

하나의 커널로는 복잡한 이미지 분석 어려워 커널 여러 개 사용, 하이퍼파라미터의 하나로써 커널 개수 정하는 일 역시 중요

### 모델구현하기

* 완전 연결 계층: 인접한 모든 뉴런과 상호 연결된 계층

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 2차원 구성으로 직관적, X의 첫차원은 입력 데이터 개수, 마지막 1은 특징 개수, 데이터가 회색조라 색깔 하나뿐이라 1
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# 출력값 10개 분류
Y = tf.placeholder(tf.float32, [None, 10])
# dropout위한 플레이스 홀더
keep_prob = tf.placeholder(tf.float32)

# 3x3 커널 가진 컨볼루션 계층 제작, 오른쪽과 아래쪽으로 한칸씩 움직이는 32개 커널 가진 컨볼루션 계층
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
# padding은 커널 슬라이딩 시 이미지의 가장 외곽에서 한칸 밖으로 움직이는 옵션
L1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
# 활성함수, 컴볼루션 완성
L1 = tf.nn.relu(L1)

# 풀링계층 만들기, 앞에 만큼 컨볼루션 계층을 입력으로 사용, 커널 크기 2x2인 풀링 계층
# strides는 승라이딩 시 두 칸씩 움직이겠다는 옵션
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 3x3커널 64개로 구성한 컨볼루션 계층과 2x2크기의 풀링 계층으로 구성
# 32는 앞서 구성한 첫번째 컨볼루션 계층의 커널 개수
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

# 추출한 특징으로 10개의 분류를 만들어내는 계층 구성
# 10개 분류는 1차원, 차원 줄이기부터! 직전 풀링이 7*7*64이므로 그크기의 1차원 계층 제작
# 배열 전체를 최종 출력값의 중간 단계인 256개 뉴럼으로 연결하는 신경망 제작
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
# 과적합 막기 dropout 사용
L3 = tf.nn.dropout(L3, keep_prob)

# 직전 은닉층 출력값 256개 받아 0~9출력 값인 10개 출력값 제작
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
# 추후 optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)를 써서 비교해보자
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 학습과 결과 코드 작성, 앞과 큰 차이 없고 모델에 입력값을 전달하기 위해 28x28로 재구성 하는 부분만 좀 달라
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        
        _,cost_val = sess.run([optimizer, cost],
                             feed_dict = {X:batch_xs,
                                         Y:batch_ys,
                                         keep_prob:0.7})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1),
         'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                      feed_dict={X:mnist.test.images.reshape(-1, 28, 28, 1),
                                Y:mnist.test.labels,keep_prob: 1}))
```

### 고수준 API

layers 모듈을 활용해서 코드를 간단하게 줄일 수 있다.

### 더보기

학습시간 꽤 걸린다, 컴퓨터 자원 필요하다, 좋은 컴퓨터도 있지만 클라우드 컴퓨팅 이용하는 방법도 있다. Cloud ML을 이용해 매우 편하게 학습시킬 수 있다. 복잡한 모델 하고 싶다면 확인 해보라

## Autoencoder

지도학습 : 프로그램에게 원하는 결과를 알려주고 학습하게 하는 방법, X와 Y둘다 있는 상태에서 학습

비지도학습 : 입력값으로부터 데이터의 특징을 찾아내는 학습 방법, 이 방법중 가장 널리 쓰이는게 오토인코더, 비지도 학습은 X만 있는 상태에서 시작

### 오토 인코더 개념

입력값과 출력값을 같게 하는 신경망, 가눙데 계층의 노드 수가 입력값보다 적어, 입력 데이터 압축효과, 노이즈 제거에 효과적, 입력층으로 들어온 데이터를 인코더를 통해 은닉층으로 내보내고, 은닉층의 데이터를 디코더를 통해 출력층으로 내보낸 뒤 만드렁진 출력값을 입력값과 비슷해지도록 만드는 가중치 찾는 것, 변이형, 잡음제거 등 다양한 형태 있다.

### 구현하기

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 파라미터 미리 설정해서 코드 구조화
learning_rate = 0.01
traing_epoch = 20
batch_size = 100
n_hidden = 256
n_input = 28*28

# 비지도라 Y없음
X = tf.placeholder(tf.float32, [None, n_input])

# 인코더 제작, n_hidden 개의 뉴런 가진 은닉층 제작, 가중치, 편향 원하는 설정
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
# sigmoid 활성 함수 사용, n_input < n_hidden
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# 인코더와 같은 구성, 입력 값이 은닉층 크기 출력값이 입력층 크기
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

# 가중치 최적화 위한 손실함수 제작, 출력값과 입력갑을 가장 비슷하게 만드는게 목적
# 압축된 은닉층의 뉴런들로 입력값 특징 알아내기 가능
# 디코더가 내보낸 결괏값과의 차이를 손신값, 값의 차이로 거리 함수 구하기
cost = tf.reduce_mean(tf.pow(X - decoder,2))

# RMSPropOptimizer 함수 활용 죄적화 함수 설정
optimizer = tf. train.RMSPropOptimizer(learning_rate).minimize(cost)

# 학습 코드
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(traing_epoch):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict = {X:batch_xs})
        total_cost += cost_val
        
    print('Epoch:', '%04d' % (epoch + 1),
         'Avg. cost =', '{:.4f}'.format(total_cost/total_batch))
    
print('최적화 완료!')

# 결과를 정확도 아닌 디코더로 생성해낸 결과를 직관적 방법으로 확인, 10개 테스트 데이터 가벼와 디코더 이용 출력값 제작
sample_size = 10
sample = sess.run(decoder, feed_dict={X:mnist.test.images[:sample_size]})

# numpy 사용 MNIST 데이터를 28x28크기 이미지 데이터로 재구성 후 출력
fig, ax = plt.subplot(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test,images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))
    
plt.show()
```

## GAN

2016년 가장 뜨거운 감자, 딥러닝의 미래 GAN!

* GAN: 결과물을 생성하는 모델중 하나, 서로 대립하는 두 신경망 경쟁시키며 결과물 생성 방법 학습, 위조지폐범과 경찰의 싸움 같아 최종적으로 위조지폐범은 진짜와 구분 불가한 위조지폐만든다!
* 구분자: 이미지를 주고 진짜인지 판단
* 생성자: 노이즈로 임의 이미지 제작

구분자는 실제 이미지, 생성자 이미지를 받아 진짜 인지 판단하게 하고 생성자는 임의의 이미지를 만들어 구분자에게 계속 제출한다. 생성자는 구분자가 구분할 수 없는 이미지를 만드는 것을 목표로 하고 구분자는 생성자가 만드는 이미지를 전부 가짜라고 구분하는 것을 목표로 한다. -> 경쟁을 통해 결과적으로 생성자는 실제 이미지와 상당히 비슷한 이미지 생성가능

고흐 풍 그림 그리기, 선으로 그려진 만화 자동채색, 모자이크 제거 등 놀라운 결과, 현재는 자연어 문장 생성 등에 관한 연구 중

### 기본 모델 구현

GAN 학습은 loss_D와 loss_G 둘 모두 최대화 시키는 것, 서로 연관 되어 두 손실값이 항상 같이 증가하는 경향을 보이지는 않는다. 경쟁관계

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot = True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28 * 28
n_noise = 128

X = tf.placeholder(tf.float32, [None, n_input])
# 비지도 학습이므로 Y 사용 없음, 구분자에 넣을 이미지가 실제 이미지와 생성 이미지 두개
# 가짜 이미지는 노이즈에서 생성하므로 노이즈를 입력할 플레이스 홀더 Z추가
Z = tf.placeholder(tf.float32, [None, n_noise])

# 신경망에 사용할 변수 설정, 가중치와 편향은 은닉층으로 출력하기 위한 변수 들
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
# 두번째 가중치과 편향은 출력층에 사용할 변수들
# 출력층에서 쓸 것이기에 실제 이미지 크기와 같아야 한다.
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 신경망에 사용할 변수 설정, 은닉층은 생성자와 동일, 구분자는 얼마나 진짜 같나 0~1로
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden],stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# 생성자 신경망 구성
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output

# 구분자 신경만 구성
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output

# 무작위 노이즈 제작 유틸리티 함수
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# 노이즈 Z이용해 가짜 이미지 만들 생성자 G
# G, X 넣어 진짜인지 팝별하게 한다.
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

# 손실값, 이번에는 두개 필요
# 생성자가 만든 이미지가 가짜라고 판단하게 하는 손실값과 진짜라고 판단하게 하는 손실값
# 진짜 라고 판별하는 D_realdl 1에 가까울수록 성공적, D_gene이 0에 가까울 수록 성공적
# 코드 간단 경찰학습
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))
# 위조지폐범 학습
loss_G = tf.reduce_mean(tf.log(D_gene))

# 학습 제작, loss_D,loss_G 각각 구분자, 생성자 신경망 각각의 변수만 써야 한다.
# 그래야 다른 것 학습 할때 상대것 안바껴
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 논문 따르면 GAN은 loss를 최대화 해야 하지만 minimize밖에 없으니 - 붙인다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

# 학습 코드 작성, 두개를 학습 해야 하므로 코드 추가
# 세션 설정
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 결괏값을 받을 변수 설정, 미니배치로 학습 반복, 구분자는 X값을, 생성자는 노이즈인 Z값 받는다.
total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0
for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)
        
        _, loss_val_D = sess.run([train_D, loss_D], feed_dict = {X:batch_xs, Z:noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z:noise})
    print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))
    if epoch == 0 or (epoch + 1) % 10 ==0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z:noise})
        
        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))
        
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
        
        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)),bbox_inches='tight')
        plt.close(fig)
        
print('최적화 완료!')
```

### 원하는 숫자 생성하기

원하는 숫자를 생성하는 모델 만들기, 다양한 방법 중 노이즈에 레이블 데이터를 힌트로 넣어주는 방법 사용

손실함수 GAN 논문 방식과는 달라, http://bamas.hithub.io/2016/08/09/deep-completion/ 참고

TODO:제대로 안나온다 나중에 다시 하자

## RNN

RNN:순환 신경망, 이미지에 CNN이라면 자연처리는 RNN 순서가 있는 데이터를 처리하는데 강점, 전후 단어 상태로 전체 의미가 달라지거나 앞의 정보로 다음의 정보 추측할 때 성능 좋다. 구글의 신경망 기반 기계번역, 수개월 만에 기존 통계기반 뛰어 넘어, 몇몇 언어에서는 인간에 가까운 수준

### MNIST를 RNN으로

손글씨 이미지를 RNN방식으로 학습하고 예측하는 모델 만들어 보자

연산하는 것을 셀이라고 하며 셀을 중첩하여 심층 신경망 제작, 앞 단계 학습 결과를 다음 단계의 학습에 이용, 학습 데이터를 단계별로 구별해서 넣어야 한다. 위에서 아래로 쓰는경우가 많으니 그렇게 입력

RNN을 직접 구현하려면 매우 복잡한 계산을 거쳐야 한다. 텐서플로우 이용하면 간단, 다양한 방법 제공, 긴간뎨의 데이터를 학습 할때 맨 뒤에서 맨 앞에 정보 잘 기억 못해, 보완하기 위한 다양한 구조 생성, 그중 많이 쓰이는게 LSTM, GRU는 LSTM보다 구조 간단. 

```python
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

# n_step 차원 추가
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# BasicLSTMCell, GRUCell 등 다양한 방식의 셀 제공, 직접 구현하려면 다름 신경망 보다 복잡한 계산식,
# 저수준부터 하려면 다른 신경망 보다 복잡한 계산 필요
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

# 셀, 입력값, 자료형 만으로 간단히 신경망 생성 가능
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# 최종 출력값을 만들어 보자, 원핫 인코딩이므로 손실함수는 tf.nn.softmax_cross_entropy_with_logits_v2사용
# RNN 출력 값은 각 단계가 포함된 [batch_size, n_step, n_hidden] 형태로 출력
# dynamic_rnn 함수 옵션중 time_major를 True로 하면 [n_step, batch_size, n_hidde]형태로 출력
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 신경망 학습하고 결과 확인하는 코드 작성 할 것, 앞장 코드와 비슷 입력따라 데이터 형태 바껴
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))
        
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys})
        
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy, feed_dict={X:test_xs, Y:test_ys}))
```

### 단어 자동 완성

단어 자동 완성 프로그램을 만들어 보자, 영문자 4개 단어 학습시켜 3개 입력시 하나 추천하는 프로그램, dynamic_rnn의 sequence_length 쓰면 가변 길이 단어 학습 가능, 짤은 단어는 가장 긴 단어의 길이 만큼 뒷부분을 0으로 채우고, 해당 단어의 길이를 계산해 squence_length로 넘겨 주면 된다. 일단 고정길이

```python
# 알파벳 순서에서 인덱스를 원-핫 인코딩으로 취한다.
import tensorflow as tf
import numpy as np

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
           'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z']
num_dic = {n:i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 학습용 단어
seq_data = ['word', 'wood', 'deep', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

# 학습에 사용할 수 있는 형식으로 변환해주는 유틸리티 함수 작성
# 입력값으로 단어의 세글자 알파벳 인덱스를 구한 배열제작
# 출력용으로 마지막 글자의 알파벳 인덱스 
# 입력값 원-핫 인코딩 변환
# 실측값 인코딩 않고 그대로 사용, 손실 ㅎ마수로 다른 것 사용, 자동 변환
def make_batch(seq_data):
    input_batch = []
    target_batch = []
    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)
        
    return input_batch, target_batch

# 신경망 구현
learning_rate = 0.01
n_hidden = 128
total_epoch = 30
n_step = 3
n_input = n_class = dic_len

# 첫 3글자만 단계적 학습하므로 n_step은 3
# 본격적 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

# 원-핫 아닌 인덱스 그대로 사용, 값 하나뿐인 1차원 배열 입력
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 두개 RNN 셀 생성, 심층 신경망 제작, DropoutWrapper로 과적합 방지
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# 함수 조합하여 심층 순환 신경 망 제작
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, status = tf.nn.dynamic_rnn(multi_cell, X, dtype = tf.float32)

# 출력층 제작
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

# 손실함수와 최적화로 구성 마무리
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 신경망 학습 코드, make_batch 이용, seq_data에 저장한 입력, 실츨 분리 최적화 실해
sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y: target_batch})
    
    print('Epoch:', '%04d'%(epoch +1), 'cost = ', '{:.6f}'.format(loss))
    
print('최적화 완료!')

# 예측 단어를 정확도와 함께 출력 하도록 만들자
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

# 예측 모델 돌려보자
input_batch, target_batch = make_batch(seq_data)
predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X:input_batch, Y:target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)
    
print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)
```

### Sequence to Sequence

* Sequence to Sequence: 구글이 기계번역에 사용하는 신경망 모델, 번역이나 챗봇 등 문장을 입력 받아 다른 문장을 출력하는 프로그램에서 사용, 인코더로 원문 읽기, 디코더로 번역 결과물 받기, 디코더가 출력한 결과물을 번역 결과 물과 비교하며 학습 