# bri = [input(), input(), input()]
# bri[0], bri[2] = bri[2], bri[0]
# dp = []
# for i in bri[2]:
#     dp.append([0,0])
# dp.append([1,1])
# for i in range(len(bri[0])):
N=list(map(int, input().split(' ')))
result=0
if N[0]>2*N[1]:
    if (N[0] - 2 * N[1])>N[2]:
        print(N[1])
    else:
        