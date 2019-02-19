T = int(input())
def get_queen(chess_map, cnt, N, x, y):
    for i in range(N):
        if x - i >= 0 and y - i >=0:
            chess_map[cnt][x - i][y - i] = True
        if x + i < N and y + i < N:
            chess_map[cnt][x + i][y + i] = True
        if x - i >= 0 and y + i < N:
            chess_map[cnt][x - i][y + i] = True
        if x + i < N and y - i >=0:
            chess_map[cnt][x + i][y - i] = True
    for i in range(N):
        if x - i >= 0:
            chess_map[cnt][x - i][y] = True
        if x + i < N:
            chess_map[cnt][x + i][y] = True
        if y - i >= 0:
            chess_map[cnt][x][y - i] = True
        if y + i < N:
            chess_map[cnt][x][y + i] = True


def sol(chess_map, cnt, N, x, y, ans):
    for i in range(N):
        for j in range(N):
            chess_map[cnt][i][j] = chess_map[cnt - 1][i][j]
    if chess_map[cnt][x][y] == False:
        get_queen(chess_map, cnt, N, x, y)

for test_case in range(1, T + 1):
    N = int(input())
    chess_map = [[[False] * N for _ in range(N)] for i in range(N)]
    ans = [0]
    sol(chess_map, 1, N, 0, 0, ans)