for test_case in range(1,11):
    ans = 0
    N = int(input())
    a = list(map(int, input().split()))
    i = 2
    while i < N - 2:
        b = [0, 0, 0, 0]
        b[0] = a[i] - a[i - 2]
        b[1] = a[i] - a[i - 1]
        b[2] = a[i] - a[i + 1]
        b[3] = a[i] - a[i + 2]
        j = 3
        while j > 0:
            if b[j] < b[j - 1]:
                b[j], b[j - 1] = b[j - 1], b[j]
            j -= 1
        i += 1
        if b[0] > 0:
            ans += b[0]
    print('#'+str(test_case)+' '+str(ans))