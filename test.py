star = [
    '*',
    '* *',
    '*****'
]
space = [
    ' ',
    '   ',
    '     '
]
def hall(n):
    if n==1:
        return 5
    else:
        return 2 * hall(n - 1) + 1
N=24#int(input())
for i in range(N):
    print(' ' * (N - 1 - i), end='')
    for j in range((i // 3) + 1):
        if (i//3)-1==j:
            print(space[i % 3], end='')
        else:
            print(star[i % 3], end='')
        print(' '*(5 - 2 * (i%3)), end='')
    print()