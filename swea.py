numbers = []
i = 0
while i <=1000000 : 
    numbers += [0]
    i += 1
i = 1
while i <=1000000 :
    j = i
    while j <= 1000000 :
        numbers[j] += i
        j += i
    i += 2
i = 1
while i <=999999 :
    numbers[i + 1] += numbers[i]
    i += 1
T = int(input())
for test_case in range(1, T + 1):
    N = input()
    N1 = N.split(' ')
    ans = numbers[int(N1[1])] - numbers[int(N1[0])-1]
    print(f'#{test_case} {ans}')
