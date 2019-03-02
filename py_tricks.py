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



# the get() method on dicts and its "default" argument

name_for_userid= {382: 'Alice', 590: 'Bob', 951: 'Dilbert'}


def greeting(userid):
    print(f'Hi {name_for_userid.get(userid, "there")}!')

greeting(590)
greeting(4242)


# using namedtuple is way shorter than defining a class manually

from collections import namedtuple

Car = namedtuple('Car', 'color mileage')

# our new car class works as expected:

my_car = Car('red', 3812.4)
my_car.color
my_car.mileage
my_car
my_car.color = 'blue'
my_car
type(my_car)
my_car

Point = namedtuple('Point', ['x', 'y', 'z'])

p1 = Point(11, 12, 13)
p1
p1.x


# zen of python
import this

zen_of_python = '''
The Zen of Python, by Tim Peters
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
'''


























