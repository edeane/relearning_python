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




def cheesshop(kind, *args, **kwargs):
    print('-- Do you have any', kind, '?')
    print("-- I'm sorry we're all out of", kind)
    for arg in args:
        print(arg)
    print('-' * 40)
    for kw in kwargs:
        print(kw, ':', kwargs[kw])

cheesshop('chedar', 'a', 'b', 'c', 'd', 'e', a=1, b=2, c=3)

# left off here https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists


















































