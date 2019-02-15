for test_case in range(1, 11):
    dump = int(input())
    li = list(map(int, input().split()))
    li_height = [0] * 101
    max_ptr = 0
    min_ptr = 100
    for i in li:
        li_height[i] += 1
        if i > max_ptr:
            max_ptr = i
        if i < min_ptr:
            min_ptr = i
    while dump > 0:
        li_height[max_ptr] -= 1
        li_height[max_ptr - 1] += 1
        li_height[min_ptr] -= 1
        li_height[min_ptr + 1] += 1
        if li_height[max_ptr] == 0:
            max_ptr -= 1
        if li_height[min_ptr] == 0:
            min_ptr += 1
        dump -= 1
    print(f'#{test_case} {max_ptr - min_ptr}')