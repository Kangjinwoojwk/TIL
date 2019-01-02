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
    if bonus_number in my_numbers:
        cnt+=2
print(pri[cnt])
#1등 :my_numbers==real_numbers
#2등 :5개+보너스
#3등 :real&my 5개 같다
#4등 :real&my 4개 같다
#5등 :real&my 3개 같다
#꽝