card=input()
chk = [0]*10
for i in card:
    chk[int(i)] += 1
ch = 2
i = 0
while i < 10:
    if chk[i] >= 3:
        chk[i] -= 3
        ch -= 1
    i += 1
i = 0
while i < 8:
    if chk[i]!=0 and chk[i] == chk[i + 1] == chk[i + 2]:
        ch -= chk[i]
        chk[i] = chk[i + 1] = chk[i+2] = 0
    i += 1
if ch == 0:
    print('baby-gin')
else:
    print('not baby-gin')