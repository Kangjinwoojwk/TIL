import requests
import random

numbers=list(range(1,46))
my_numbers=random.sample(numbers,6)
my_numbers.sort()
#my_numbers=[2,6,25,30,33,45]

url='https://www.nlotto.co.kr/common.do?method=getLottoNumber&drwNo=837'
response = requests.get(url, verify=False)
lotto_data=response.json()
real_numbers=[]
for key, value in lotto_data.items():
    if 'drwtNo'in key:
        real_numbers.append(value)
real_numbers.sort()
bonus_number = lotto_data['bnusNo']
# my_numbers=[1, 2, 3, 4, 5, 6]
# real_numbers=[1, 2, 3, 5, 6, 7]
# bonus_number=4
#print(bonus_number)
# print(my_numbers)
# print(real_numbers)
#my_numbers, real_numbers, bonus_number

cnt=len(set(my_numbers)&set(real_numbers))
if cnt==5:
    if bonus_number in my_numbers:
        cnt+=2
if cnt==6 : 
    print("1등")
elif cnt==7 :
    print("2등")
elif cnt==5 :
    print("3등")
elif cnt==4 :
    print("4등")
elif cnt==3 :
    print("5등")
else:
    print("꽝")
#1등 :my_numbers==real_numbers
#2등 :5개+보너스
#3등 :real&my 5개 같다
#4등 :real&my 4개 같다
#5등 :real&my 3개 같다
#꽝