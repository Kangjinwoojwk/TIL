T = int(input())
for test_case in range(T):
    a, b = list(map(int, input().split()))
    n = b - a
    chk = 0
    i = 1
    count = 0
    while chk < n:
        count += 1
        chk += i
        if count % 2:
            continue
        else:
            i += 1
    print(count)