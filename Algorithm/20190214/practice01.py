data=[7, 4, 2, 0, 0, 6, 0, 7, 0]
fall_max = 0
for i in range(len(data)):
    cnt = 0
    for j in range(i + 1, len(data)):
        if data[j] >= data[i]:
            cnt += 1
    else:
        if (len(data) - i - 1 - cnt) > fall_max:
            fall_max=(len(data) - i - 1 - cnt)
print(fall_max)