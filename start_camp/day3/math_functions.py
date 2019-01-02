print(__name__)

def average(numbers):
    return sum(numbers)/len(numbers)
def cube(x):
    return x * x * x
def main():
    my_score = [79, 84, 66, 93]
    print(my_score)
    print('평균 : ',average(my_score))
    print(cube(3))
if __name__=='__main__':
    main()