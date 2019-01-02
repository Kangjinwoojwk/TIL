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