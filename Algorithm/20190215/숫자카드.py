T = int(input())
for test_case in range(1, T+1):
    n = input()
    N = input()
    l = [0] * 10
    for i in N:
        l[int(i)] += 1
    max_v = 0
    for i in l:
        if max_v < i:
            max_v = i
    i = 9
    while i >= 0:
        if l[i] == max_v:
            break
        i -= 1
    print(f'#{test_case} {i} {l[i]}')