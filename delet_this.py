i = ['1']
ii = (1,1,1)
iii = [[1, 1, [1, 1, 1]], [1, 1, 1]]

def stuff():
    try:
        round('a', 2)
    except Exception:
        raise Exception


def a():
    try:
        stuff()
    except Exception:
        raise Exception
    
def b():
    try:
        a()
    except Exception:
        raise Exception

iiii = [1,2,3,4,5,6]

iiiii = "a"
# b = iiiii.split('.')

try:
    print(i)
    b()
    # print(0/0)
except Exception:
    raise Exception

print("stuff")
