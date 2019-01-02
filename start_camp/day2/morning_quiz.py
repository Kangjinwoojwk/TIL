#1. 평균을 구하시오
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
    print(city,':', avg_temp)
    #print('{0} : {1}'.format(city,avg_temp))

#4:도시 중에 최근 3일간 가장 추웠던 곳, 가장 더웠던 곳
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

#all_temp로 모은다.
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