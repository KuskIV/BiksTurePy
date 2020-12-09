# import logging, sys
# import traceback

# i = ['1']
# ii = (1,1,1)
# iii = [[1, 1, [1, 1, 1]], [1, 1, 1]]

# def stuff():
#     try:
#         round('a', 2)
#     except Exception as e:
#         print(f"ERROR: {e}")
#         raise Exception('this is shocking')


# def a():
#     try:
#         stuff()
#     except Exception:
#         raise Exception
    
# def b():
#     try:
#         a()
#     except Exception:
#         raise Exception

# iiii = [1,2,3,4,5,6]

# iiiii = "a"
# # b = iiiii.split('.')

# try:
#     print(i)
#     b()
#     # print(0/0)
# except Exception:
#     print("a")
# else:
#     print("else bois")
# finally:
#     print("done bois")

# print("stuff")


if __name__ == '__main__':
    import re
    
    test_string1 = 'fog1'
    test_string2 = 'Original10'
    
    # noise = re.search('[A-Za-z]+', test_string2)
    # number = re.search('[0-9_-]+', test_string2)
    
    noise = re.search('[\D]+', test_string2)
    number = re.search('[\d]+', test_string2)
    
    
    
    if re.match('(o|O)riginal', noise.group(0)):
        print('yeet')
    
    print(noise.group(0))
    print(number.group(0))
    