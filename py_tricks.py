"""
Python Tricks Newsletter from Real Python
https://realpython.com/python-tricks/
"""

from pprint import pprint

# merge two dictionaries

x = {'a': 1, 'b': 2}
y = {'b': 3, 'c': 4}

z = {**x, **y}
pprint(z)


# why python is great
# function argument unpacking

def myfunc(x, y, z):
    print(x, y, z)

tuple_vec = (1, 0, 1)
dict_vec = {'x': 1, 'y': 0, 'z': 1}

myfunc(*tuple_vec)
myfunc(**dict_vec)

# lambda func
# shortcut for declaring small and anonymous functions

add = lambda x, y: x + y
add(5, 3)

def add(x, y):
    return x + y

add(4, 3)
(lambda x, y: x + y)(6, 7)

# different ways to test multiple flags at once in Python

x, y, z = 0, 1, 0

if x == 1 or y == 1 or z == 1:
    print('passed')

if 1 in (x, y, z):
    print('passed')

if x or y or z:
    print('passed')

if any((x, y, z)):
    print('passed')


# sort a python dict by value (get a list)

xs = {'a': 4, 'b': 3, 'c': 2, 'd': 1}
xs

idx_one = lambda x: x[1]
sorted(xs.items(), key=idx_one, reverse=True)

from operator import itemgetter

sorted(xs.items(), key=itemgetter(1), reverse=True)



























