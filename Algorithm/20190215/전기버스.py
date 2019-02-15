T = int(input())
for test_case in range(1, T+1):
    K, N, M = list(map(int, input().split()))
    M_stop = list(map(int, input().split()))
    M_stop += [0]
    count = 0
    bus = N
    chk = N
    i = M - 1
    while i >= -1:
        if bus - M_stop[i] <= K:
            chk = M_stop[i]
        else:
            if bus == chk:
                count = 0
                break
            bus = chk
            i += 1
            count += 1
        i -= 1
    print(f'#{test_case} {count}')