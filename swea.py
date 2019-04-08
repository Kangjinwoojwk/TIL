def fuction(a, b, c):
    def sol(x):
        return a * (x ** 2) + b * x + c
    return sol
a = fuction(1, 2, 3)
for i in range(10):
    print(a(i))