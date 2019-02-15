T=int(input())
for test_case in range(1, T+1):
    N, M = list(map(int, input().split()))
    num_list = list(map(int, input().split()))
    max_sum = 0
    min_sum = 2 ** 31 - 1
    i = 0
    while i <= N - M:
        sum_v = 0
        j = i
        while j < i + M:
            sum_v += num_list[j]
            j += 1
        if sum_v > max_sum:
            max_sum=sum_v
        if sum_v < min_sum:
            min_sum=sum_v
        i += 1
    print(f'#{test_case} {max_sum-min_sum}')