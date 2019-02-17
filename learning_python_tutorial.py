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



















