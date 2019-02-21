for test_case in range(10):
    test_num = int(input())
    arr = [input() for _ in range(100)]
    ans = 0
    for i in range(100):
        for j in range(100):
            x = 1
            y = 1
            while i - x >= 0 and i + x < 100:
                if arr[i - x][j] == arr[i + x][j]:
                    x += 1
                else:
                    if 2 * (x - 1) + 1 > ans:
                        ans = 2 * (x - 1) + 1
                    break
            while j - y >= 0 and j + y < 100:
                if arr[i][j - y] == arr[i][j + y]:
                    y += 1
                else:
                    if 2 * (y - 1) + 1 > ans:
                        ans = 2 * (y - 1) + 1
                    break
            x = 1
            y = 1
            while i - x >= 0 and i + x - 1 < 100:
                if arr[i - x][j] == arr[i + x - 1][j]:
                    x += 1
                else:
                    if 2 * (x - 1) > ans:
                        ans = 2 * (x - 1)
                    break
            while j - y >= 0 and j + y - 1 < 100:
                if arr[i][j - y] == arr[i][j + y - 1]:
                    y += 1
                else:
                    if 2 * (y - 1) > ans:
                        ans = 2 * (y - 1)
                    break
    print(f'#{test_num} {ans}')