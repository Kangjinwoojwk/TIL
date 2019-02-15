T = int(input())
for test_case in range(1, T + 1):
    N = int(input())
    l = list(map(int,input().split()))
    min_v = 1000001
    max_v = 0
    i = 0
    while i < N:
        if l[i] < min_v:
            min_v=l[i]
        if l[i] > max_v:
            max_v=l[i]
        i += 1
    print(f'#{test_case} {max_v-min_v}')