# 다리 만들기
# 여러 섬으로 이루어진 나라가 있다. 이 나라의 대통령은, 섬들을 잇는 다리를 만들겠다는 공약으로 인기몰이를 해 당선될 수 있었다. 하지만 막상 대통령에 취임하자, 다리를 놓는다는 것이 아깝다는 생각을 하게 되었다. 그래서 그는, 생색내는 식으로 한 섬과 다른 섬을 잇는 다리 하나만을 만들기로 하였고, 그 또한 다리를 가장 짧게 하여 돈을 아끼려 하였다.
# 이 나라는 N×N크기의 이차원 평면상에 존재한다. 이 나라는 여러 섬으로 이루어져 있으며, 섬이란 동서남북으로 육지가 붙어있는 덩어리를 말한다. 다음은 세 개의 섬으로 이루어진 나라의 지도이다.
# 위의 그림에서 색이 있는 부분이 육지이고, 색이 없는 부분이 바다이다. 이 바다에 가장 짧은 다리를 놓아 두 대륙을 연결하고자 한다. 가장 짧은 다리란, 다리가 격자에서 차지하는 칸의 수가 가장 작은 다리를 말한다. 다음 그림에서 두 대륙을 연결하는 다리를 볼 수 있다.
# 물론 위의 방법 외에도 다리를 놓는 방법이 여러 가지 있으나, 위의 경우가 놓는 다리의 길이가 3으로 가장 짧다(물론 길이가 3인 다른 다리를 놓을 수 있는 방법도 몇 가지 있다).
# 지도가 주어질 때, 가장 짧은 다리 하나를 놓아 두 대륙을 연결하는 방법을 찾으시오.
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]


def getmap(x, y, n):
    global N
    get_map[x][y] = n
    for i in range(4):
        if 0 <= x + dx[i] < N and 0 <= y + dy[i] < N and input_map[x + dx[i]][y + dy[i]] == 1 and get_map[x + dx[i]][y + dy[i]] == 0:
            getmap(x + dx[i], y + dy[i], n)
    
def sol(x, y, n, distance):
    global ans
    if distance >= ans:
        return
    if get_map[x][y] != 0 and get_map[x][y] != n:
        ans = distance
        return
    for i in range(4):
        if 0 <= x + dx[i] < N and 0 <= y + dy[i] < N and get_map[x + dx[i]][y + dy[i]] != n:
            sol(x + dx[i], y + dy[i], n, distance + 1)



N = int(input())
input_map = [list(map(int, input().split())) for _ in range(N)]
get_map = [[0] * N for _ in range(N)]
island = 1
ans = 2 * N
for i in range(N):
    for j in range(N):
        if get_map[i][j] == 0 and input_map[i][j] == 1:
            getmap(i, j, island)
            island += 1
for i in range(N):
    for j in range(N):
        if get_map[i][j] != 0:
            sol(i, j, get_map[i][j], 0)
print(ans - 1)