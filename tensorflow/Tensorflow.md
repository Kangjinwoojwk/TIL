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