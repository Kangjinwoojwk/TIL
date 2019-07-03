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