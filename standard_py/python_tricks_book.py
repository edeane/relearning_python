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
# "is" vs "=="
# "is" if point to same object
# == if equivalent values

a = [1, 2, 3]
b = a

a == b
a is b
b is a

c = a[:]
a == c # True
a is c # False not the same memory

# __repr__ and __str__
# add __repr__ to every class
#

class Car(object):
    def __init__(self, color, mileage):
        self.color = color
        self.mileage = mileage

    # def __str__(self):
    #     return f'a {self.color} car'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!r}, {self.mileage!r})'


my_car = Car('red', 100_000)
print(my_car) # returns __str__ dunder
str(my_car)
f'{my_car}'
my_car # returns __repr__ dunder
[my_car]
str(my_car)
repr(my_car)

import datetime

today = datetime.date.today()
str(today)
repr(today)

# 4.3 Defining Your Own Exception Classes

class NameTooShortError(ValueError):
    pass


def validate(name):
    if len(name) < 10:
        raise NameTooShortError(name)


validate('eddie')


# 4.4 Cloning Objects for Fun and Profit
# Assignment statements in Python do not create copies
# just bind names to an object
# shallow copy = won't create copy of child objects
# deep copy = creates copy of all child objects

orig_list = [1, 2, 3]
orig_dict = {'a': 1,
             'b': 2,
             'c': 3}
orig_set = {1, 2, 3}

new_list = list(orig_list)
new_dict = dict(orig_dict)
new_set = set(orig_set)

hex(id(orig_list))
f'0x{id(orig_list):x}'
f'{id(new_list):x}'


xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
ys = list(xs)
xs.append(['new sublist'])
xs
ys
xs[1][0] = 'X'


import copy

xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# copy.copy() is shallow copy
# copy.deepcopy is deep copy

zs = copy.deepcopy(xs)
zs
xs.append(['new sublist'])
xs[1][0] = 'X'
xs
zs


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point({self.x!r}, {self.y!r})'


a_p = Point(23, 42)
a_p
b_p = copy.copy(a_p)
b_p
a_p is b_p
a_p == b_p


class Rectangle(object):
    def __init__(self, topleft, bottomright):
        self.topleft = topleft
        self.bottomright = bottomright

    def __repr__(self):
        return f'Rectangle({self.topleft!r}, {self.bottomright!r})'


rect = Rectangle(Point(0, 1), Point(5, 6))
srect = copy.copy(rect)
rect
srect

rect.topleft.x = 999
rect
srect

drect = copy.deepcopy(rect)
drect
rect.bottomright.y = 422
rect
drect
srect


# 4.5 Abstract Base Classes Keep Inheritance in Check
# help make class hierarchies easier to maintain

from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    @abstractmethod
    def foo(self):
        pass

    @abstractmethod
    def bar(self):
        pass


class Concrete(Base):
    def foo(self):
        pass

    def bar(self):
        pass


issubclass(Concrete, Base)

c = Concrete()



# 4.6 Namedtuples
# immutable, fix the issue of selecting the wrong index
# write once, ready many
# a memory efficent shortcut ot defining an immutable class manually
# _ named methods and attributes are to avoid collision of

from collections import namedtuple
import json

Car = namedtuple('Car', ['color',
                         'mileage',
                         'year',
                         'make',
                         'model',
                         ])

my_car = Car('orange', 100_000, 1999, 'ford', 'mustang')
my_car.make
my_car.model
color, mileage, _, _, _ = my_car
color
mileage
print(*my_car)

{my_car: 'abc123'}
json.dumps(my_car._asdict())


# class vs instance variables

class Dog(object):
    num_legs = 4

    def __init__(self, name):
        self.name = name

jack = Dog('jack')
jill = Dog('jill')
jack.num_legs
jill.num_legs
# jack.num_legs = 6
Dog.num_legs = 8


# 4.8 Instance, Class, and Static Methods!


class MyClass:

    # normal instance method that can change the self instance of the class
    # can also modify self.__class__ attribute
    def method(self):
        return 'instance mehthod called', self

    # points to cls paramter not self
    # modifies all instances of the class
    @classmethod
    def classmethod(cls):
        return 'class method called', cls

    # primarily for namespacing your methods
    # cannot modify object state or class
    @staticmethod
    def staticmethod():
        return 'static method called'



obj = MyClass()
obj.method()
obj.classmethod()
obj.staticmethod()

# error because python cannot call the self instance
MyClass.method()
MyClass.classmethod()
MyClass.staticmethod()

import math

class Pizza:

    def __init__(self, radius, ingredients):
        self.radius = radius
        self.ingredients = ingredients

    def __repr__(self):
        return f'Pizza({self.radius!r}, {self.ingredients!r})'

    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])

    @classmethod
    def prosciutto(cls):
        return cls(['mozzarella', 'tomatoes', 'ham'])

    def area(self):
        return self.circle_area(self.radius)

    @staticmethod
    def circle_area(r):
        return r ** 2 * math.pi




custom_pizza = Pizza(12, ['mozzarella', 'tomatoes', 'ham'])
custom_pizza
custom_pizza.area()
custom_pizza.circle_area(12)


m_pizz = Pizza.margherita()
m_pizz
m_pizz.ingredients
Pizza.prosciutto().ingredients

m_pizz.circle_area(18)


# 5.1 Dictionaries, Maps, Hashtables
# strings and number keys
# now dicts are ordered as of 3.7 due to CPython, but better to be explicit and use OrderedDict if order is needed

phonebook = {
    'bob': 7387,
    'alice': 3719,
    'jack': 7052,
}


squares = {x: x * x for x in range(6)}

phonebook['alice']
squares[3]

from collections import OrderedDict, defaultdict, ChainMap

# read only dict
from types import MappingProxyType


d = OrderedDict(one=1, two=2, three=3)
d = OrderedDict([('one', 1), ('two', 2), ('three', 3)])
d
d.keys()

dd = defaultdict(int)
dd
s = 'mississippi'
for k in s:
    dd[k] += 1

dd['p']
dd['r']


dict1 = {'one': 1,
         'two': 2}

dict2 = {'three': 3,
         'four': 4}

chain = ChainMap(dict1, dict2)

chain['two']
chain['five']


read_only = MappingProxyType(dict1)
read_only['one']
read_only['three'] = 3



# 6.7 Iterator Chains



def integers():
    for i in range(1, 9):
        yield i

def squared(seq):
    for i in seq:
        yield i * i

def negated(seq):
    for i in seq:
        yield -i

integers_g = range(1, 9)
squared_g = (i * i for i in integers_g)
negated_g = (-i for i in squared_g)


chain = negated(squared(integers()))
next(chain)

for i in chain:
    print(i)

for i in negated_g:
    print(i)



# 7 Dictionary Tricks

# 7.1 Dictionary Default Values

name_for_userid = {
    382: 'Alice',
    950: 'Bob',
    590: 'Dilbert',
}

def greeting(userid):
    return f'Hi {name_for_userid.get(userid, "there")}'


greeting(950)
greeting(4242)

# 7.2 Sorting Dictionaries for Fun and Profit
# Press Ctrl + P to see parameter info
# https://www.jetbrains.com/help/pycharm/viewing-reference-information.html

xs = {'a': 4, 'c': 2, 'b': 3, 'd': 1}
sorted(xs.items())
sorted(xs.items(), key=lambda x: x[1], reverse=True)

import operator

xs.items()

sorted(xs.items(), key=operator.itemgetter(1))


sorted(xs.items(), key=lambda x: abs(x[1]))


# 7.3

def handle_a(a, b):
    return a + b

def handle_b(a, b):
    return (a * a) + (b * b)

def handle_default(a, b):
    return (a * 2) + (b * 2)

func_dict = {
    'cond_a': handle_a,
    'cond_b': handle_b,
}

conds = ['cond_a', 'cond_b', 'cond_c']
for cond in conds:
    print(func_dict.get(cond, handle_default)(3, 4))


def dispatch_dict(operator, x, y):
    return {
        'add': lambda: x + y,
        'sub': lambda: x - y,
        'mul': lambda: x * y,
        'div': lambda: x / y,
    }.get(operator, lambda: None)()

dispatch_dict('mul', 2, 8)


# 7.4 The Craziest Dict Expression in the West

xs = {True: 'yes', 1: 'no', 1.0: 'maybe'}
xs


xs = dict()
xs[True] = 'yes'
xs[1] = 'no'
xs[1.0] = 'maybe'
xs

True == 1 == 1.0

['no', 'yes'][True]
['no', 'yes'][False]
'abc'[True]

(hash(True), hash(1), hash(1.0))

xs = {'a': 1, 'b': 2}
ys = {'b': 3, 'c': 4}


zs = {**xs, **ys}
zs

# Pretty Printing Dicts

from pprint import pprint
import json

print(json.dumps(zs, indent=4, sort_keys=True))
pprint(zs, indent=1, width=-1, depth=4)


mapping = {'a': 23, 'b': 42, 'c': 12648430, 'd': set([1, 2, 3])}
pprint(mapping, width=1)

import datetime

pprint(dir(datetime), width=-1)
pprint(dir(datetime.datetime), width=-1)
pprint(dir(datetime.date), width=-1)


pprint([_ for _ in dir(datetime) if 'date' in _.lower()], width=-1)
help(datetime.date.fromtimestamp)


def greet(name):
    return 'Hello, ' + name + '!'

greet.__code__
greet.__code__.co_consts
greet.__code__.co_varnames

# with dis.dis we can see the easier to read representation of the bytecode

import dis

dis.dis(greet)






















