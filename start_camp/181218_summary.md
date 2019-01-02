# 181218 수업정리

## 1. 개발환경 설정

- chocolatey
  - 패키지 다운로드
- python v3.6.7
  - 개발언어
- git
- VS code
  - 실제 코디용  
- chrome
  - 인터넷 검색용
- typora
  - 마크업 툴

## 2. 디렉토리

* VS 컨+`로 터미널 열기
* cd ~: user
* mkdir "   ":디렉토리 만들기
* touch "   ":파일 만들기
* rmdir "   ":디렉토리 지우기 

## 3. 파이썬

* 우선 TIL에 파이썬 파일 생성
  - python "    ".py:파이썬 파일 실행
  - python:파이썬버전 정보 등
* 파이썬에서 제공하는 형식들
  * int, bool 등
  * 리스트 등
  * 2중 리스트를 통해 2차원 배열 가능, []안에[]-2차원 안에 3차원 등등

## 4.리스트

* list 추출하기

  1.리스트에 접근하기

  2.자료 잘라내기

  - 리스트 선언시 바로 연산 가능, 인덱스는 마이너스도 받는다. 가장 끝이 -1 좌로 올때마다 작아져
  - 타입캐스팅, 레인지는 레인지일뿐 앞에 list가 list로 바꿔 주는 것
  - numbers[시작수:끝수]:시작은 포함 끝은 미포함

  리스트안에 딕셔너리를 넣을 수 있다 이 경우[숫자][태그]로 찾을수 있다.

  ``` python
  numbers=[1,2,3]  #변수 이름은 뜻을 담아서 짓자!
  family = ['mom', 1.64,'dad',1.75,'sister',1.66]
  mcu=[
      ['ironman','captain','vision','hulk'],
      ['xmen','deadpool'],
      ['spiderman']
  ]
  disney=mcu[0]
  disney[2]
  mcu[0][2]
  numbers=list(range(100))
  numbers[1:10]
  numbers[2:5]#[start:end] start 포함, end 미포함
  x=['life', 'is', 'short', True, 123, ['you','need', 'python']]
  print(x[-3])
  ```

  deadpool에 접근하려면?

* 리스트끼리 더하면 리스트가 합쳐진다.

  - +=연산으로 추가만 할 수 있다.

* del(team[2])등 사용으로 삭제 가능

## 5.딕셔너리

* 딕셔너리는 태그를 사용하여 데이터에 접근 할 수 있다.

* ```python
  my_info = {
      'name':'neo',
      'job':'hacker',
      'e-mail':'kjw03230@nate.com',
      'mobile':'010-****-1713',
      'date of birth':'1991.03.23'
  }
  my_info['e-mail']
  
  hphk=[
      {
          'name':'John',
          'email':'John@hphk.io'
      },
      {
          'name':'neo',
          'email':'neo@hphk.io'
      },
      {
          'name':'tak',
          'email':'tak.hphk.io'
      }
  ]
  hphk[2]['email']
  ```

  deadpool에 접근하려면?

* 리스트끼리 더하면 리스트가 합쳐진다.

  - +=연산으로 추가만 할 수 있다.

* 대부분의 키는 스트링, 이에 대해 어떻게 받고 어떻게 접근하는지가 문제

  dhkj[1]['name'][4]['family_name'] 등 무궁무진한 형태

## 6.Function 함수

* 소괄호가 있으면 다 함수

* 자주 발생하는 문제 한번만 코딩하자....

* max(리스트):리스트중 제일 큰거, round()사사오입, ceil-천장, round(수,정수)-뒷수만큼의 소수자리점 아래서 반올림

* 함수 설명, 공식문서가 재미없어도 제일 정확python round function documentation

  * https://docs.python.org/3/library/functions.html
    * help()메소드 괄호안에 그냥은 필수, 대괄호 안은 옵션
  * 출처별로 신뢰도 달라

* ```python
  first=[11.25,18.,20.0]
  second=[10.75,9.50]
  full=first+second
  full_sorted=sorted(full)
  reverse_sorted=sorted(full,reverse=True)
  ```

## 7.메소드

* 점이 찍혀 있는 것, ㅡㅡ_.ㅡㅡ_() 이것들도 함수

* 메소드 별로 사용법 다르고 출력 달라, 어떤 건 결과를 뱉고 어떤건 안뱉어 대신 원래거만 정렬된다던지

* Object가 할 수 있는 함수가 메소드이다.

* ```python
  my_list=[4,7,9,1,3,6]
  max(my_list)
  min(my_list)
  #특정 요소의 인덱스?
  my_list.index(7)
  #리스트를 뒤집으려면?
  my_list.reverse()
  dust=100    #int, 100은 있어도 다른정수는 개념 선언된건 오브젝트
  language='python'
  samsung=['elec','sds','s1']
  samsung.index('sds')
  lang='python'
  lang.capitalize()
  lang.replace('on','off')
  
  samsung.append('bio')#원본이 바뀐다! 넣는거 안들어간다!
  ```

  메소드별 출력이 달라 외우기보다 익숙해져야한다.

| str      | int  | list          | bool          |
| -------- | ---- | ------------- | ------------- |
| 'python' | 100  | ['1','2','3'] | true or false |

## 8.웹

* 웹브라우저를 포함시켜서 페이지 열어보기

* url뜯어보기 실제로 우리가 필요한 것은 굉장히 적어, url 바꾸는 것

* ```python
  import webbrowser
  
  keywords=[
      '삼성전자 주가',
      '스마트 팩토리',
      '나가사와 마리나',
      '애로우 시즌7',
      'the 100 s6'
  ]
  for keyword in keywords:
      url='https://www.google.com/search?q='+keyword
      webbrowser.open_new(url)
  ```

* ```python
  import requests
  
  url='https://www.nlotto.co.kr/common.do?method=getLottoNumber&drwNo=837'
  response = requests.get(url, verify=False)
  print(response.text)
  ```

* 받은 것은 일단 스트링, 딕셔너리 아냐, 파싱 필요, 응답과 데이터는 달라

* ```python
  import requests
  
  url='https://www.nlotto.co.kr/common.do?method=getLottoNumber&drwNo=837'
  response = requests.get(url, verify=False)
  lotto_data=response.json()
  real_numbers=[]
  for key in lotto_data:  #key는 그냥 쓴거 i, a,c 다 상관없음
      if'drwtNo' in key:
          real_numbers.append(lotto_data[key])
  print(real_numbers)
  
  for key, value in lotto_data.items():
      if 'drwtNo'in key:
          real_numbers.append(value)
  
  # real_numbers=[
  #     lotto_data['drwtNo1'],
  #     lotto_data['drwtNo2'],
  #     lotto_data['drwtNo3'],
  #     lotto_data['drwtNo4'],
  #     lotto_data['drwtNo5'],
  #     lotto_data['drwtNo6']
  # ]
  #print(real_numbers)
  ```

* API는 보통 제이슨으로 준다. JSON viwer chrome을 써서 보자

* 로또 몇등 당첨됐는지 확인하는 코드가 과제로!