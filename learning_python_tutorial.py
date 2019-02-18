# -*- coding: utf-8 -*-
'''
Going through this https://docs.python.org/3/tutorial/index.html
'''

# ---- 3. An Informal Introduction to Python ----

spam = 1
text = "# This is not a comment because it's inside quotes"
# this is a comment
2+2
50 - 5*6
(50-5*6)/4
8/5
17/3

17 // 3 # floor division

17 % 3 # remainder

5 * 3 + 2

5 ** 2 # power

2** 7

width = 20
height = 5 * 9
width * height


4 * 3.75 - 1

tax = 12.5 / 100
price = 100.5
price * tax
price + _ # add previous number
round(_, 2)



# strings

'spam eggs'

# escape with \
'doesn\'t'
"doesn't"

s = 'First line.\nSecond line.'
print(s)

# dont want special characters you can use raw strings
print('C:\some\name')
print(r'C:\some\name')

print('''\
Usage: thingy [OPTIONS]
    -h                          Display this usage message
    -H hostname                 Hostname to connect to
''')


'3 + 3 ' + '= 6'
3 * 'un' +'ium'

'Py' 'thon'


text = 'put sever stings with parenthses ' \
       'to have the joined togeteher'

text = ('put several string within parentheses '
        'to have them joined together.')

text


prefix = 'Py'
prefix + 'thon'

word = 'Python'
word[0]
word[2:4]
word[-2]
word[-2:]

how_indecies_work = '''\
 +---+---+---+---+---+---+
 | P | y | t | h | o | n |
 +---+---+---+---+---+---+
 0   1   2   3   4   5   6
-6  -5  -4  -3  -2  -1
'''

print(how_indecies_work)

len(word)


# lists
squares = [1, 4, 9, 16, 25]

squares[-1]
squares[-3:]
# copy of the list
squares[:]

squares + [36, 49, 64, 81, 100]

# immutable
# - strings
# - tuple
# - integer
# - bool

# mutable
# - lists
# - sets
# - dictionaries


cubes = [1, 8, 27, 65, 125]

cubes[3] = 64
cubes

cubes.append(6**3)
cubes.append(7 ** 3)
cubes


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
len(letters)


a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
x
x[1][1:]

# Fibonacci series:
a, b = 0, 1
while a < 100:
    print(a, end=',')
    a, b = b, a+b

i = 256*256

print('The value of is is', i)

a_list = ['abc', 'def', 'ghi']
print(a_list, 'this is a string', 'this is another string', sep=', ')



# ---- 4. More Control Flow Tools ----

x = int(input('Please enter an integer: '))

if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('one')
else:
    print('more')


# measure some strings:

words = ['cat', 'window', 'defenestrate']

for w in words:
    print(w, len(w))


for word in words[:]:
    if len(word) > 6:
        words.insert(0, word)

words



for i in range(5):
    print(i)


for j in range(5, 10):
    print(j)

for k in range(0, 10, 3):
    print(k)


for l in range(-10, -100, -30):
    print(l)

a = ['Mary', 'had', 'a', 'little', 'lamb']

for i in range(len(a)):
    print(i, a[i])

for i, j in enumerate(a):
    print(i, j)

# creates lists from iterables
list(range(5))

# break, continue, pass

# break = breaks out of innermost
for n in range(2, 100):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        print(n, 'is a prime number')

# continue = continues to next iteration
for num in range(2, 10):
    if num % 2 == 0:
        print('found an even number', num)
        continue
    print('found an odd number', num)

# pass = does nothing
def initlog(*args):
    '''
    This is an example function. Remember to implement this!

    Args:
        *args:

    Returns: nothing yet

    '''
    pass

# google docstrings
# http://google.github.io/styleguide/pyguide.html
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

def fib(n):
    '''Print a Fibonaccie series up to n.

    Args:
        n (int): Up to n

    Returns:
        None
    '''
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b


fib(n=100)
fib(2000)


fib

# all functions return None
print(fib(100))

def fib2(n):
    '''Return a list containing the Fibonacci series up to n.

    Args:
        n (int): Up to n
    Returns:
        List with Fibonacci series
    '''
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result


fib_one_hundred = fib2(100)
type(fib_one_hundred)
fib_one_hundred

fib_one_hundred = [str(i) for i in fib_one_hundred]
fib_one_hundred
' '.join(fib_one_hundred)


def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)



ask_ok('are you ok?')




def cheeseshop(kind, *args, **kwargs):
    '''Prints out cheeseshop stuff out

    Args:
        kind (str): kind of cheese
        *args: any args to pass through
        **kwargs: any named args
    Returns:
        None
    '''
    print('-- Do you have any', kind, '?')
    print("-- I'm sorry we're all out of", kind)
    for arg in args:
        print(arg)
    print('-' * 40)
    # for kw in kwargs:
    #     print(kw, ':', kwargs[kw])
    for key, value in kwargs.items():
        print('key:', key, 'value:', value)


cheeseshop('chedar', 'a', 'b', 'c', 'd', 'e', a=1, b=2, c=3)


def concat(*args, sep=''):
    '''Simple concat function with any separator

    Args:
        *args(str): any list of strings
        sep(str): the value used for between strings

    Returns:
        Concatenation of strings with sep between values.
    '''
    return sep.join(args)


concat('first quant', 'second quant', 'final quant', sep=' | ')


def parrot(voltage, state='a stiff', action='voom'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")

parrot(10)

d = {'voltage': 'four million',
     'state': 'bleedin demised',
     'action': 'VOOM'}

# unpacking argument lists
parrot(**d)



# lambda functions are small anaonymous functions


def make_incrementor(n):
    '''Returns a addition function.

    Args:
        n(int): starting value
    Returns:
        Function which accepts a value to increment.
    '''
    return lambda x: x + n


forty_two_inc = make_incrementor(42)
forty_two_inc(7)
type(forty_two_inc)


pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]

pairs.sort(key=lambda x: x[0])

# CamelCase for classes
# and lower_case_with_underscores for functions and methods


# 5. Data Structures

# Lists

test_list = [1, 2, 3, 'a', 'b', 'c']

test_list.append(4)
test_list.extend(['d', 'e', 'f'])
test_list.insert(1, '11')
test_list.remove('11')
test_list.remove(4)
# remove and return
test_list.pop(2)
test_list.clear()
test_list = [1, 2, 3, 'a', 'b', 'c', 1, 1, 'a', 'a']

# returns index of value
test_list.index('b')
test_list.count('a')
test_list.pop(1)

len(test_list)
test_list.reverse()

# these are the same ways you can copy a list
test_list.copy() == test_list[:]

test_list


# stacks

stack = [3, 4, 5]
stack.append(6)
stack.append(7)
stack.pop()

stack


# deque from collections

from collections import deque

queue = deque(['Eric', 'John', 'Michael'])
queue.append('Terry')
queue.append('Graham')
queue.popleft()
queue.pop()
queue



# list comprehensions

squares = []

for x in range(10):
    squares.append(x**2)

squares

squares = list(map(lambda x: x**2, range(11)))
squares = [x**2 for x in range(112)]

squares


[(x, y) for x in [1,2,3] for y in [1,3,4] if x !=y]


from math import pi


[str(round(pi, i)) for i in range(10)]


freshfruit = ['  banana', '  loganberry  ', 'passion fruit   ']

# strip strings
freshfruit
[weapon.strip() for weapon in freshfruit]


matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]


matrix

[[row[i] for row in matrix] for i in range(4)]

list(zip(matrix, 'abc'))


list(zip(*matrix))

x = [1, 2, 3]
y = [4, 5, 6]

list(zip(x, y))

# unpack lists to arguments
args = [2, 7]
list(range(*args))

# delete items in list or entire list
a = [-1, 1, 66.25, 333, 333, 1234.5]
del a[0]
a
del a


# immutable tuples

t = (123, 456, 789)
t[1] = 756

x, y, z = t
x
y
z


# sets
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)

'orange' in basket


a = set('aaabbbdcccdddeee')
a

{x for x in 'abracadabra' if x not in 'abc'}




# dictionaries


tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127

tel

sape = 'sape'
del tel[sape]
tel
'jack' in tel
'guido' not in tel

tel_dict = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
tel_dict
{x: x**2 for x in (2, 4, 6)}
{x: x**2 for x in range(2, 10, 2)}

knights = {'gallahad': 'the pure', 'robin': 'the brave'}


for i, j in knights.items():
    print(i, j)


# enumerate
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)



# zip again

questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']


for q, a in zip(questions, answers):
    print('What is your {0}? It is {1}'.format(q, a))


for i in reversed(range(1, 10, 2)):
    print(i)

basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']

sorted(set(basket))

for f in sorted(set(basket)):
    print(f)

import math

raw_data = [56.2, float('NaN'), 51.7, 55.3, 52.5, float('NaN'), 47.8]
raw_data
filtered_data = []

for value in raw_data:
    if not math.isnan(value):
        filtered_data.append(value)

filtered_data

(1, 2, 3) < (1, 2, 4)
(1, 2, 3) == (1.0, 2.0, 3.0)


# 6. Modules


import os
os.getcwd()

import fibo

fibo.a_dict
fibo.a_integer
fibo.a_list


fibo.fib(10)
fibo.fib2(10)

for i in sys.path:
    print(i)

import sys

for i in dir(fibo):
    if i[0] not in '_':
        print(i)

dir()

import builtins

for bi in dir(builtins):
    print(bi)


# Packages = a collection of modules

sound_package_example = """\
sound/                          Top-level package
      __init__.py               Initialize the sound package
      formats/                  Subpackage for file format conversions
              __init__.py
              wavread.py
              wavwrite.py
              aiffread.py
              aiffwrite.py
              auread.py
              auwrite.py
              ...
      effects/                  Subpackage for sound effects
              __init__.py
              echo.py
              surround.py
              reverse.py
              ...
      filters/                  Subpackage for filters
              __init__.py
              equalizer.py
              vocoder.py
              karaoke.py
              ...
"""

print(sound_package_example)

# 7. Input and Output

year = 2016
event = 'Referendum'

f'Results of the {year} {event}'

yes_votes = 42_572_654
no_votes = 43_132_495
yes_votes
type(yes_votes)

percentage = yes_votes / (yes_votes + no_votes)
percentage
'{:,} YES votes {:2.2%}'.format(yes_votes, percentage)

# format string syntax https://docs.python.org/3/library/string.html#format-string-syntax

# left off here https://docs.python.org/3/tutorial/inputoutput.html#fancier-output-formatting



str(123) # human readable
repr(123) # read by the interpreter

s = 'Hello, world.'
str(s)
repr(s)
str(1/7)
x = 10 * 3.25
y  =200 * 200

s = 'The value of x is ' + repr(x) + ', and y is ' + repr(y) + '...'
s
print(s)

hello = 'hello, world\n'
hellow = repr(hello)
hellow
print(hellow)

repr((x, y, ('spam', 'eggs')))

# formatted string literals

import math

print(f'The value of pi is approximately {math.pi:.3f}')

table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
for name, phone in table.items():
    print(f'{name:10} ==> {phone:10d}')


# !a applies ascii(), !s applies str, and !r applies repr

animals = 'eels'
print(f'My hovercraft is full of {animals}.')
print(f'My hovercraft is full of {animals!r}.')
print(f'My hovercraft is full of {animals!s}.')



animals_dict = {'bird': 'fly', 'fish': 'swim', 'human': 'walk', 'bear': 'strong', 'leopard': 'run', 'snail': 'slow'}

print('bird: {0[bird]}, fish: {0[fish]}, human: {0[human]}'.format(animals_dict))
print('bird: {bird}, fish: {fish}, human: {human}'.format(**animals_dict))


# should use the formatted string literals over '.format()' if possible

print('{0} and {1}'.format('spam', 'eggs'))
print('The story of {0}, {1}, and {other}'.format('Bill', 'Manfred', other='Georg'))



for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))


start_n, stop_n = 1, 12
max_char = len(str((stop_n - 1) ** 3))
for x in range(start_n, stop_n):
    print(f'{x:{max_char}} {x*x:{max_char}} {x*x*x:{max_char}}')


for i in range(start_n, stop_n):
    print(repr(i).rjust(2), repr(i**2).rjust(3), repr(i**3).rjust(4))


# zero pad
'347'.zfill(4)
'-3.14'.zfill(7)
print(math.pi)
'3.1415926535897'.zfill(5)

# reading and writing files
# 'r' read only
# 'w' write over
# 'a' appending
# 'b' binary
# \n on unix \r\n on windows
# use with so it is properly closed

with open('requirements.txt') as f:
    require_data = f.read()

print(require_data)
require_data_lst = require_data.split('\n')
require_data_lst.pop()
require_data_lst

req_dict = dict()
for i in require_data_lst:
    pack, ver = i.split('==')
    req_dict[pack] = ver

for i, j in req_dict.items():
    print(i, j)

with open('new_file_written.txt', 'w') as f:
    f.write('This is a test\nthis is another line')


# can dump objects
import json

json.dumps([1, 'simple', 'list'])

with open('json_dump', 'w') as f:
    json.dump([animals_dict, req_dict], f)

with open('json_dump', 'r') as f:
    x = json.load(f)

x

# can also use pickle
# pickle data coming from an untrusted source can execute arbitrary code it the data was crafted by
# a skilled attacker


# 8. Errors and Exceptions



while True:
    try:
        x = int(input('Please enter a number: '))
        break # break out of the while loop
    except ValueError:
        print('Oops! That was not a valide number. Try again...')



a_file = 'requirements.txt'

try:
    f = open(a_file, 'r')
except OSError:
    print('cannot open', a_file)
else:
    print(a_file, 'has', len(f.readlines()), 'lines')
    f.close()

try:
    raise Exception('spam', 'eggs')
except Exception as inst:
    print(inst)


def this_failes():
    x = 1/0


try:
    this_failes()
except Exception:
    print('error')


try:
    this_failes()
except Exception as inst:
    print('error:', inst)


raise NameError('HiThere')



# Classes!!! attributes and methods

class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

    def f(self, x):
        return x**3


comp_class = Complex(3.0, -4.5)

comp_class.r
comp_class.i
comp_class.f(3)


# class Dog:
#     tricks = [] # incorrect because it will be shared by all instances
#     def __init__(self, name):
#         self.name = name
#     def add_trick(self, trick):
#         self.tricks.append(trick)

class Dog:
    def __init__(self, name):
        self.name = name
        self.tricks = []
        self._private = 42

    def add_trick(self, trick):
        self.tricks.append(trick)

    def print_private(self):
        print(self._private)


a = Dog('fido')
b = Dog('buddy')

a.add_trick('walk')
a.add_trick('run')
a.tricks
b.tricks
b.print_private()
b.print_private()

# Iterators use iter() and next()

s = 'abc'
it = iter(s)

next(it)
next(it)


class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

rev_spam = Reverse('green eggs and ham')

for char in rev_spam:
    print(char)


# Generators use yield

def reverse_fun(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]


for char in reverse_fun('golf'):
    print(char)


# sum of squares
sum(i**2 for i in range(1, 10))

data='golf'
[data[i] for i in range(len(data)-1, -1, -1)]


# 15. Floating Point Arithmetic: Issues and Limitations


# base 2, base 10, and base 16
0.125

a_num = 0.125

str(a_num)
[bin(int(i)) for i in str(a_num) if i not in '.']

bins = ['0', '1', '10', '11', '100', '101', '110', '111', '1000', '1001', '1010']
[int(a_bin, 2) for a_bin in bins]


int('0b1010', 2)
bin(10)
int(bin(10), 2)

2 ** 0
2 ** 1
2 ** 2
2 ** 3
2 ** 4



an_int = 10
the_bins = []


def convert_int_to_base(n=10, base_to=2):
    """Converts number from base to new base

    Args:
        n(int): an integer
        base_to(int): base to convert to

    Returns:
        String with new base number
    """
    n = int(n)
    bins = []
    while n > 0:
        n_bin = n % base_to
        bins.append(n_bin)
        n = int(n / base_to)

    return ''.join(map(str, bins[::-1]))

def conv_frac_to_base(n=.625, base_to=2):
    n = n - int(n)
    pass

n = 10.625
n = n - int(n)
n

# loop
''.join(str(x) for x in reversed(the_bins))

convert_base(17)
convert_base(10, 4)

from time import sleep

def print_convert_base(n_from=1, n_to=10, base_to=2, n_sleep=1):
    """

    Args:
        n_from(int): from int
        n_to(int): to int

    Returns:
        None
    """
    from_pad = len(str(n_to))
    to_pad = len(convert_base(n_to))

    for i in range(n_from, n_to+1):
        new_i = convert_base(i, base_to=base_to)
        print(f'{i:>{from_pad},} to base 2: {new_i:>{to_pad}}')
        sleep(n_sleep)


print_convert_base(10_000, 10_100, base_to=2, n_sleep=.5)


.625
1/2 + 0/4 + 1/8


int(13 / 2)
int(_ / 2)
1 % 2


.625 / .5
.625 % .5
.125 % .5
.125 / .5
.25 / .5
.25 % .5



