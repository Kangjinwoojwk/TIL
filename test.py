N=int(input())
get=[]
for i in range(N):
    get.append(input())
ans = 0
for i in get:
    chk=''
    chk1=set()
    for j in i:
        if (chk!=j) and (j not in chk1):
            if chk=='':
                pass
            else:
                chk1.add(chk)
            chk=j
        elif chk==j:
            pass
        else:
            break
    else:
        ans += 1
print(ans)