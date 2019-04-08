# def this():
#     def sol():
#         return 'this'
#     return sol()
#
#
# these = this()
# print(these)

# def my_func(arg):
#     return arg
my_func = lambda arg: arg

def fco():
    return lambda n: n + 1

# print(fco)
num_100 = 100
num_101 = fco()(num_100)
print(num_101)

my_list = [1, 2, 3, 4]
[1, 2, 3, 4][0]



# def arg_func():
#     print('i`m function')
#
#
# my_func(arg_func())
# my_func(arg_func)()