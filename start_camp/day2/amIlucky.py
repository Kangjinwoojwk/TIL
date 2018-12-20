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
    