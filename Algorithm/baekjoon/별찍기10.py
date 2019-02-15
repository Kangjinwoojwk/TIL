# 나중에 더 효율적으로 바꿔보자
def sol(x, y, result, N):
    n = N//3
    for i in range(n, 2 * n):
        for j in range(n, 2 * n):
            result[x + i][y + j] = ' '
    if n <= 1:
        return
    for i in range(3):
        for j in range(3):
            if i==j==1:
                continue
            sol((x + (i * n)), (y + (j * n)), result, n)
N = int(input())
result = []
for i in range(N):
    result.append(['*']*N)

sol(0, 0, result, N)
for i in range(N):
    for j in range(N):
        print(result[i][j], end='')
    print()




# print 의 문제가 아니다. 다른 로직이 필요하다
# N = int(input())
# chk = []
# i = 1
# while i <= N:
#     chk += [i]
#     i *= 3
# result = []
# for i in range(N):
#     result.append('')
# for i in range(N):
#     for j in range(N):
#         k = len(chk) - 1
#         while k > 0:
#             if (i % chk[k]) // chk[k - 1] == (j % chk[k]) // chk[k - 1] == 1:
#                 result[i] += ' '
#                 break
#             k -= 1
#         else:
#             result[i] += '*'
# for i in result:
#     print(i)


# 답은 맞는데 시간 초과, print가 너무 많아서 그런 듯
# N = int(input())
# chk = []
# i = 1
# while i <= N:
#     chk += [i]
#     i *= 3
# for i in range(N):
#     for j in range(N):
#         k = len(chk) - 1
#         while k > 0:
#             if (i % chk[k]) // chk[k - 1] == (j % chk[k]) // chk[k - 1] == 1:
#                 print(' ', end='')
#                 break
#             k -= 1
#         else:
#             print('*', end='')
#     print()