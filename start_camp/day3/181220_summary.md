# 181220 학습일지

## 1. 과제 이야기

HTML과 CSS, 과제 나갈것, CSS는 강의할만한건 아냐, 코딩이라기보단 디자인, 해볼수록 늘어

h1은 왜 있는가? 헤드설정? 뼈대 구성을 위함 h1:가장 큰 제목, 등장 한번뿐, 가장 중요한 것이기 때문에 크게 보여주기 위해서 커진 것 뿐

div 예전엔 모두 div로 묶어, 공간...그러다 섹션, 헤더 등 역할위주로 이름 바뀌기 시작해, 마크업은 그냥 역할 지정일뿐, 브라우저가 멋대로 크게 보여주는 것뿐 브라우저세팅에서 바꾸면 크기 등 바껴, 크기 작다고 h? 안돼, 스타일 시트에서 하는 것

스타일 시트를 잘 못다루면...?br이 가진 의마, 한줄 띄기가 아니라 다른 기능, 다들 기능이 달라, 이렇게 썼더니 이렇게 보이더라에서 온 것, 왜 그런지 다시 봐야돼, CSS는 그리기 도구, 프론트엔드와 백엔드

오늘,  받기만 했는데 API처럼 던지는걸 만든다, API서버를 짤 것, 클라이언트->서버

### 로또 코딩

기존에 있는 함수랑 다른 부분 만들기

```python
import requests
import random

numbers=list(range(1,46))
url='https://www.nlotto.co.kr/common.do?method=getLottoNumber&drwNo=837'
response = requests.get(url, verify=False)
lotto_data=response.json()
real_numbers=[]
for key, value in lotto_data.items():
    if 'drwtNo'in key:
        real_numbers.append(value)
real_numbers.sort()
bonus_number = lotto_data['bnusNo']
cnt=0
count=0
pri={
    6:'1 등',
    7:'2 등',
    5:'3 등',
    4:'4 등',
    3:'5 등',
    2:'꽝',
    1:'꽝',
    0:'꽝'
}
while cnt!=6:
    #간추린코드 python라이브러리 사용형
    my_numbers=random.sample(numbers,6)
    cnt=len(set(my_numbers)&set(real_numbers))
    count+=1
    if cnt==5:
        if bonus_number in my_numbers:
            cnt+=2
    print(pri[cnt])
    print('      ',count)
    #원래 코드 C++이식형
    # cnt=0
    # my_numbers=random.sample(numbers,6)
    # count+=1
    # for i in my_numbers:
    #     for j in real_numbers:
    #         if i==j:
    #             cnt+=1
    # if cnt==5:
    #     for i in my_numbers:
    #         if bonus_number==i:
    #             cnt+=2
    # if cnt==6 : 
    #     print("1등")
    #     print('      ',count)
    # elif cnt==7 :
    #     print("2등")
    # elif cnt==5 :
    #     print("3등")
    # elif cnt==4 :
    #     print("4등")
    # elif cnt==3 :
    #     print("5등")
    # else:
    #     print("꽝")
    
```

세트...순서가 없어, []으로 접근 못해, 있는지는 물을 수 있어, 중복불가, 합집합, 차집합은 알 것

함수 제작, def를 써서 정의, 함수는 모두 어딘가에 저장되어 있을 것이다. 호출과 정의, 앞에 미리 정의해줘야 수행한다. 

range는 기본 정수단위로 움직여 1단위,비어 있는건 none, 리턴은 없을수도 있어, 동작과 무관, sort()는 아무것도 리턴안해 함수 제작

```python
import requests
import random

def pick_lotto():
    numbers = random.sample(range(1,46),6)
    numbers.sort()
    return numbers
def get_lotto(turn):
    url = 'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo='
    url += str(turn)
    response = requests.get(url)
    lotto_data = response.json()
    numbers=[]
    for key, value in lotto_data.items():
        if 'drwtNo'in key:
            numbers.append(value)
    numbers.sort()
    bonus_number = lotto_data['bnusNo']
    return {
        'numbers' : numbers,
        'bonus' : bonus_number
    }
    
def am_i_lucky(get_numbers,draw_numbers):
    cnt=len(set(get_numbers)&set(draw_numbers['numbers']))
    pri={
        6:'1 등',
        7:'2 등',
        5:'3 등',
        4:'4 등',
        3:'5 등',
        2:'꽝',
        1:'꽝',
        0:'꽝'
    }
    if cnt==5:
        if draw_numbers['bonus'] in get_numbers:
            cnt+=2
    return pri[cnt] 
result = am_i_lucky(pick_lotto(),get_lotto(837))
print(result)
```

## 2.오후 수업

* 리팩토링-다른 사람들과 같이 쓰자! 보기만 해도 느낌 오게 쓰긴했는데...변수를 바꾼다? 더 좋게?  코드를 깔끔하게 다듬고 사람들이 보기 쉽게 하는 것이 리팩토링, 기능이 바뀌는건 전부, 성능은 바뀔 수 있어, 더 잘 짠다는 것, 알기 쉽게 짠다는 것, args=arguments

* 컨벤션-관습적으로 4칸 띄는것 같은것
* 괄호를 받는 놈, 안 받는 놈, 리턴 하는 놈, 안하는놈 4종의 함수
* 함수를 전체적으로 하나의 값으로 봐주도록, 함수안에 print는 쓰지 마라, 단, 만들때 잘 되는지 확인할때 빼고
* 들어가는 인자의 형식이 바뀐다면?
* 함수를 쓰면 메인은 굉장이 짧아 질 수 있다.
* math_functions- 새로운건 없이 생각할게 있는 것
* 확장자가 같으면 import가능 대신 import한 것은 한번 돌아가는 것이므로 해당 파일에 출력이 없도록 조심한다.
* 미리 써봤던 함수들, random같은 것들도 똑같다. random.으로 쓰는 것도 똑같다.
* 함수가 전부 온다. 전부 쓸건 아닌데? 골라서 쓰고 싶다면?

```python
from math_functions import average, cube
```

* 대신 이때는 최종적으로 포함된건 함수 뿐이기에 math_function.없이 그냥 쓴다.

* 함수만 뽑아써도 원래 함수 있는 파일에 print가 있으면 출력된다.

  ``` python
  def main():
      my_score = [79, 84, 66, 93]
      print(my_score)
      print('평균 : ',average(my_score))
      print(cube(3))
  if __name__=='__main__':   #해당 파일이 main이 아니면 '__name__'은 파일명으로 나와
      main()
  ```

  main이 이 함수가 아니면 나오지 마라 

* 함수는 다른 파일에 분리 시켜놓고(모듈화), 메인은 깔끔하게 함수만 쓰는게 보기 좋다.

## 3.FLASK

* 정말 간단하게 API를 쓸 수 있게 하는 것







## 4.내일

* 텔레그램, 오토 챗봇, 하는건 할 수 있다. 재밌다곤 안했다. 힘들 것, 각오해라...