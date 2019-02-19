<<<<<<< HEAD
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
=======
ans = 0
def sol():
    global ans
    ans += 1
sol()
sol()
print(ans)
>>>>>>> 1ee5a3457a29143a7112b36e6f5201d7667f35ed
