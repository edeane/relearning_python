"""
Python Tricks: The Book
A Buffet of Awesome Python Features
- Dan Bader
https://realpython.com/products/python-tricks-book/


"""


# 2.1 Covering Your Ass with Assertions
# used for unrecoverable errors (bugs)
# for debugging not runtime errors

def apply_discount(product, discount):
    price = int(product['price'] * (1.0 - discount))
    assert 0 <= price <= product['price']
    return price

shoes = {'name': 'Fancy Shoes', 'price': 14900}

apply_discount(shoes, .25991654)
apply_discount(shoes, .25)
# apply_discount(shoes, 1.5)

# dont do this
def delete_product(prod_id, user):
    assert user.is_admin(), 'Must be admin'
    assert store.has_product(prod_id), 'Unknown product'
    store.get_product(prod_id).delete()

# use this
def delete_product(prod_id, user):
    if not user.is_admin():
        raise AuthError('Must be admin to delete')
    if not store.has_product():
        raise ValueError('Unknown product id')
    store.get_product(prod_id).delete()

assert 1 == 1, 'abc'
# always true when there is a tuple
# assert (1==2, 'abc')


# 2.2 Complacent Comma Placement

names = [
    'Alice',
    'Bob',
    'Dilbert',
    'Jane', # add comma to end
]

print(names)


# 2.3 with Statement
# properly acquiring and releasing resources
# need __enter__ (enters the context) and __exit__ (leaves the context) methods to use with in Class


class Indenter(object):
    def __init__(self):
        self.level = 0

    def __enter__(self):
        self.level += 1
        # with assigns the as name this value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.level -= 1
    
    def print(self, text):
        print('    ' * self.level + text)


a_ind = Indenter()
a_ind.print('abc')
with a_ind:
    a_ind.print('hello')
    with a_ind:
        a_ind.print('hello there')
        with a_ind:
            a_ind.print('hello you there')
    a_ind.print('hey')


with Indenter() as indent:
    indent.print('hello')
    with indent:
        indent.print('hello there')
        with indent:
            indent.print('hello you there')
    indent.print('hey')


# 2.4 Underscores, Dunders, and More
# _var = internal use
# var_ = fix naming conflict
# __var = prevents modification when inheriting objects changes to _Class__var
# __var__ = double underscore (dunder)
# _ = temporary value (for _ in range(42): print('hello')

import os
print(os.getcwd()) # relearning_python
from standard_py.my_module import *

external_func()

from standard_py.my_module import _internal_func

_internal_func()

print_ = 'this is a print variable'
print_


class Test(object):
    def __init__(self):
        self.foo = 11
        self._bar = 23
        self.__baz = 43


class ExtendedTest(Test):
    def __init__(self):
        super().__init__()
        self.foo = 'overridden'
        self._bar = 'overridden'
        self.__baz = 'overridden'

t = Test()
t2 = ExtendedTest()

from pprint import pprint


t.foo
t._bar
t._Test__baz
t2.foo
t2._bar
t2._Test__baz
t2._ExtendedTest__baz


pprint(dir(t))
pprint(dir(t2))

for _ in range(42): print('hello')
car = ('red', 'auto', 12, 3812.4)
color, _, _, mileage = car
color
_
mileage


# 2.5 string formatting

def f_greet(name, question):
    return f"Hello, {name}! How's it {question}?"

def p_greet(name, question):
    return "Hello, " + name + "! How's it " + question + "?"


f_greet('ed', 'going')
p_greet('ed', 'going')

from dis import dis

dis(f_greet) # uses the BUILD_STRING opcode as an optimization
dis(p_greet) # uses the

name = 'Eddie'
errno = 50159747054
# hex format
f"Hey {name}, there's a {errno:#x} error!"
f"Hey {name}, there's a {errno:#X} error!"


# 3.1 Everything is an object, including functions
# Objects can be made callable

def yell(text):
    return f'{text.upper()}!'


yell('hello eddie how are you')
bark = yell
bark('woof')
# functions and their names are two separate concerns
del yell
yell('hello')
bark('hello')
bark.__name__


funcs = [bark, str.lower, str.capitalize]
pprint(funcs)

for f in funcs:
    print(f('hey there'))


def greet(func):
    greeting = func('Hi, I am a Python program')
    print(greeting)


greet(bark)

# high order functions (functions that can accept other functions as arguments)
def proper(text):
    return f'{text.lower()}...'

greet(proper)


list(map(bark, ['hello', 'hi', 'hey there']))




def speak(text):

    def whisper(t):
        return t.lower()

    return f'{whisper(text)}...'

speak('Hello Python')


def get_speak_func(volume):

    def whisper(text):
        return f'{text.lower()}...'

    def yell(text):
        return f'{text.upper()}!'

    if volume > 0.5:
        return yell
    else:
        return whisper



get_speak_func(.6)('hello there World What is up')
get_speak_func(.3)('hello there World What is up')

# making a class opbject callable with the call dunder method
class Yeller(object):
    def __call__(self, text):
        return f'{text.upper()}!'

class IsThisCallable(object):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


a_yeller = Yeller()
a_yeller('omg this IS not Yelling')

a_is_this = IsThisCallable('abc')

callable(a_yeller)
callable(get_speak_func)
callable(IsThisCallable)
callable(a_is_this)
callable(a_is_this.get_name)


# 3.2 Lambda Functions
# shouldn't be used that often

add_lamb = lambda x, y: x + y
add_lamb(5, 3)

(lambda x, y: x + y)(7, 11)

list(range(-5, 6))
sorted(range(-5, 6), key=lambda x: x * x)
sorted(range(-5, 6), key=lambda x: abs(x))

[x for x in range(16) if x % 2 == 0]


# 3.2 The Power of Decorators
# wrap another function (replacing one function with another)
# allow you to modify callables
# callable that takes a callable as input and returns another callable
# require grasp of the properties of first class functions
#    - functions are objects
#    - functions can be defined inside other functions
#    - functions can capture the parent functions local state
# used for:
#    - logging
#    - enforcing access control and authentication
#    - instrumentation and timing functions
#    - rate-limiting
#    - caching and more
import time

# define wrapper
def time_it(func):

    # define new behavior
    def time_it_func(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(f'function duration: {end_time-start_time}')
        # return the result from calling the function
        return result
    # wrapper returns new function
    return time_it_func

@time_it
def long_fun(number):
    return sum(i * i for i in range(number))


long_fun(5_000_000)
long_fun(10_000_000)
long_fun(100_000_000)




def uppercase(func):
    def upper_fun(*args, **kwargs):
        result = func(*args, **kwargs).upper()
        return result
    return upper_fun


def greet(name, question):
    return f"Hello {name}, How's it {question}?"


greet('Eddie', 'going')

greet
uppercase(greet)
uppercase(greet)('Eddie', 'going')

@uppercase
def greet(name, question):
    return f"Hello {name}, How's it {question}?"

greet

# wraps carries over the docstring and other metadata of the input function
from functools import wraps

def trace(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'TRACE: calling {func.__name__}() with {args}, {kwargs}')

        result = func(*args, **kwargs)

        print(f'TRACE: {func.__name__}() returned "{result}"')

        return result

    return wrapper

@trace
def say(name, line):
    """Returns name and line concatenated"""
    return f'{name}: {line}'


say('Eddie', 'Hello there.')

say.__name__
say.__doc__


# 3.4 Fun with *args and **kwargs

def foo(req, abc=123, *args, **kwargs):
    print(req, abc)
    if args:
        print(args)
    if kwargs:
        print(kwargs)


foo('bar', 'foo', 'abc', '123', '456', eddie='deane', hello='world')


val_dict = {'ed': 'deane'}
foo('req', **val_dict)
foo('req', abc='abc', **val_dict)
val_tup = ('eddie', 'deane', 'eddie')
foo('req', *val_tup, **val_dict)

def bar(x, *args, **kwargs):
    kwargs['name'] = 'Alice'
    new_args = args + ('extra', )
    foo(x, *new_args, **kwargs)

bar('ufc', 'is', 'lit', a_name='eddie', b_name='deane')



sum([1, 7])


# 3.6 All functions return None

def print_stuff(*args):
    for arg in args:
        print(arg)
    return None

print_stuff('aasdf', 'asdf', 'oiwfjfd', 'ospiudf', 'biwer')

# Chapter 4 Classes & OOP
#













