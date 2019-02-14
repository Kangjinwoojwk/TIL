import sys
sys.stdin = open('input1.txt','r')

T = int(input())
for test_case in range(T):
    N, arr = input().split()
    N = int(N)
    print(N, arr)