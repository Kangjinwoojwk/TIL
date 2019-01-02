# 181219 수업정리

## 1.github사용법

* 초대, 22명 그룹, git다시
* git status:지금 git과 다른 것을 알려준다
* git add .:최근꺼로 찍을 준비
* git commit -m '메세지':메세지를 넣어서 git 찍기
* git push:git업로드, 처음에는 긴거 있었다
* git log:언제찍었는지 보기    



* 사이트에서 new gist 파일 공유, 깃헙에 올려서 공유

## 2.모닝퀴즈

* 평균구하기, 리스트, 딕셔너리
* sum함수를 쓰면 전부 더한 것을 쉽게 볼 수 있다.
* sum과 len을 쓰자
* 딕셔너리에서 밸루 뽑는.values()함수, 리스트가 아닌데 돈다. sum이 돌아
* ,들어가있으면 len재진다. 딕셔너리도 똑같이 코드를 좀더 친절하게 했으면

컨벤션: 안지킨다고 무너지지 않지만 다같이 지키므로인해 더 나아질거라고 생각하는 것

google style guide

* 스트링으로 변경하고 싶으면 str()
* 'city:avg_temperature'로 쓰고 싶다. ->'{0}:{1}'.format(city,avg_temperature)
* for key value in------:양손으로 꺼낸다. 앞쪽에 key, 뒤쪽에 value
* if ____ in list:list에 ___이 있으면 참 아니면 거짓

```python
1. 평균을 구하시오
my_score = [79, 84, 66, 93]
my_average = 0.0 
cnt = 0
for score in my_score:
    my_average += score
    cnt += 1
my_average /= cnt
print('    내 평균은:',my_average)

your_score={
    '수학':87,
    '국어':83,
    '영어':76,
    '도덕':100
}
your_average = 0.0 
cnt = 0

for score in your_score.values():
    your_average += score
    cnt += 1
your_average /= cnt
print('당신의 평균은:',your_average)

cities_temp={
    '서울' : [-6, -10, 5],
    '대전' : [-3, -5, 2],
    '광주' : [0, -5, 10],
    '구미' : [2, -2, 9]
}
#3:도시별 온도 평균
for city in cities_temp.keys():
    avg_temp=round(sum(cities_temp[city]) / len(cities_temp[city]),2)
    print(city,':', avg_temp) #내가 푼것
    #print('{0} : {1}'.format(city,avg_temp)) #강사님 풀이

#4:도시 중에 최근 3일간 가장 추웠던 곳, 가장 더웠던 곳, 내풀이
coldest = 100000.0
hottest = -100000.0
for city in cities_temp.keys():
    for temp in cities_temp[city]:
        if temp < coldest:
            coldest = temp
        if temp > hottest:
            hottest = temp
chk = False
print('최근 3일간 가장 추웠던 곳은',end = ' ')
for city in cities_temp.keys():
    for temp in cities_temp[city]:
        if temp == coldest:
            if chk == True:
                print(',',end = ' ')
            print(city,end = '')
            chk = True
print('입니다.')
chk = False
print('최근 3일간 가장 더웠던 곳은',end = ' ')
for city in cities_temp.keys():
    for temp in cities_temp[city]:
        if temp == hottest:
            if chk == True:
                print(',', end = ' ')
            print(city, end = '')
            chk = True
print('입니다.')

#all_temp로 모은다.강사님 풀이
all_temp=[]
for key, value in cities_temp.items():
    all_temp += value
#hottest/coldest 찾는다
hottest = max(all_temp)
coldest = min(all_temp)
for key, value in cities_temp.items():
    if hottest in value:
        hottest.append(key)
    if coldest in value:
        coldest.append(key)
print(hottest,coldest)
```

둘째 날 모닝 퀴즈, 다양한 방법, if ___in list 는 list에 있는지 리턴해준다.



## 3.Chrome

* OneTab,Momentum, C9-깃헙과 연동

* C9-집에 가면 코딩한게 없다. 어떻게 해야할까? C9, workspace, 컴퓨터 하나씩 준다

  CPU, 램, 디스크 전부 제공, 컴퓨터 분양 두번째 누르면 두번째 받아 사양 낮아도 코딩엔 지장없어

  프라이빗은 하나 밖에 안된다. 블랭크

  우분투:리눅스 기반, MS제외 대부분 리눅스 기반

  C9에서 코딩을 하면 다른 컴퓨터에서도 접속해서 코딩 가능 폰으로도 가능 git을 써서 다운,업 연속 하는 것도

  괜찮다 그러기 위해서는 우선 git에 익숙해 져야 할것

  마크다운에 익숙해질 것

  C9-  포트 폴리오 한번 만들어 볼까?

  부트 스트랩-신발 뒤의 끈? 부트스트랩으로 이쁘게~하자니 오래걸리니까 만들어진걸로다가 쓰자!

  start bootstrab view source보면 git으로넘어가

## 4. html, CSS

* 과제...www.codecademy.com
* 코드카데미-깃헙으로 접속 카탈로그, introduce HTML, 한 번 볼 것
* HTML은 태그와 태그 사이 존재, 
* clone은 가져온다는것, 다운로드랑 살짝 달라, 
* git clone (url)로 해당 내용 통째로 가져올 수 있어, 실제는 workspace만 있어
* other 폴더 못가, index로 봐바라 run후 나오는 url 을 open하여 가서 봐라
* https://others-kjw03230.c9users.io/index.html 즉각반응
* index.html이 있어야 반응한다, 이미지 크기 가능하면 맞춰라
* 컨+쉬프트+C:요소 확인, 반응형:자바스크립트의 영역
* 보는 것은 개발용. github에 올리자
* min.css는 css이 뛰어쓰기 등 다 지운 것
* git에 올려보자  c9으로 올릴땐 유저네임과 비밀번호 계속 입력 필요, 지문등록하면 안해도 돼
* github를 써서 포트폴리오를 넣자

## 과제:코드카데미