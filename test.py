bri = [input(), input(), input()]
bri[0], bri[2] = bri[2], bri[0]
count = []
count.append([1,1])
for i in bri[2]:
    count.append([0,0])
for idx1, v in enumerate(bri[0]):
    for idx2, i in enumerate(bri[2]):
        if bri[0][idx1] == i:
            count[idx2 + 1][1] += count[idx2][0]
        if bri[1][idx1] == i:
            count[idx2 + 1][0] += count[idx2][1]
print(sum(count[-1]))