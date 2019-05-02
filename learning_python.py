"""
Relearning the basics after being away from python for a year!

Comprehensive Python Cheatsheet
https://gto76.github.io/python-cheatsheet/

Learn Python in Y minutes
https://learnxinyminutes.com/docs/python3/

Python Cheatsheet
https://www.pythonsheets.com/index.html

Official Python Documentation https://docs.python.org/3/
Tutorial https://docs.python.org/3/tutorial/index.html
Standard Library https://docs.python.org/3/library/index.html
Language Reference https://docs.python.org/3/reference/index.html

Python 3 Module of the Week
https://pymotw.com/3/index.html


relearing packages order:
X pycharm stuff (venv, shortcuts, https://www.jetbrains.com/help/pycharm/meet-pycharm.html)
X python tutorial
X python standard library
X mysqlclient pandas real_sql to_sql
X unittests (https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html)
X numpy
X pandas
X plotting (mostly just seaborn)
X plotly dash
X webscraping (https://realpython.com/python-web-scraping-practical-introduction/)
- sklearn (tutorials)
X threading, parallel, multiprocessing
    X “Premature optimization is the root of all evil (or at least most of it) in programming.”)
    X CPU Bound (multiprocessing) vs I/O Bound (use asyncio when you can and threading when you must)
X py tricks book
X itertools
X creating modules and packages
X importing modules and packages https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html
- blackjack project (beat the dealer book)
- ncaabb project (https://github.com/rodzam/ncaab-stats-scraper)
- ufc stats (http://ufcstats.com/statistics/events/completed fighttrax)
- pinkbike comment sentiment
- scrape newsletters and create topics
- django
- decorators (property, classmethod, staticmethod)
- more standard library https://docs.python.org/3/library/index.html
- recursion https://realpython.com/python-thinking-recursively/

create requirements.txt
X pip freeze > requirements.txt

learn about documentation tools
X Google Docstring Standard https://github.com/google/styleguide/blob/gh-pages/pyguide.md
- https://matplotlib.org/sampledoc/
- http://www.sphinx-doc.org/en/master/
- http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html
- https://www.mkdocs.org/

learn about project folder structure
- https://drivendata.github.io/cookiecutter-data-science/ talk: http://isaacslavitt.com/2016/07/20/data-science-
    is-software-talk/
- https://blog.godatadriven.com/how-to-start-a-data-science-project-in-python

Pycharm setup setting
- Editor > Font > Size: 13
- Keymap > Execute > Execute selection in console: Ctrl + Enter
- Tools > Python External Documentation: https://stackoverflow.com/questions/49777474/does-anyone-have-the-
    documentation-urls-in-pycharm-for-the-following-libraries
- Install  https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysql-python mysqlclient
- pip install C:/Users/<username> /Downloads/mysqlclient-1.3.13-cp37-cp37m-win_amd64.whl
- Check Settings > Appearance & Behavior > Widescreen tool window layout
- Pycharm documentation ctrl + q for documentation

Conda Environment vs Python Virtual Environment
- conda: type activate relearning_python in Pychamr Terminal
- venv: already activated...

"""

# CTRL + Q for documentation
print('hello')

tasks = {'completed': 0, 'name': 'this new motion', 'ed': 'deane'}
# Basic completion CTRL + Space once
print(tasks['ed'])

# CTRL + Space twice
# sock

# Smart Completion = CTRL + Shift + Space
def f(x):
    x.append(10)
    x.remove(10)
    return x


class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
        self.sit_pos = False

    def run(self, miles):
        for i in range(miles):
            print('ran', i, 'mile')
        return miles ** 2

    def __repr__(self):
        return self.name

    def do_sit(self):
        self.sit_pos = not self.sit_pos

    def is_sit(self):
        if self.sit_pos == True:
            position = 'sitting'
        elif self.sit_pos == False:
            position = 'standing'
        return position


n = input('Enter a number: ')
print(f'{n} * 3 = {n * 3}')
n = int(n)
print(f'{n} * 3 = {n * 3}')

'123'.isdigit()
'123'.isnumeric()

n = input('Enter a number: ')
if n.isdigit():
    n = int(n)
    print(f'{n} * 3 = {n * 3}')

n = 'abc'
int(n)
try:
    n = int(n)
except ValueError:
    print(f"Can't convert {n} to integer")

import random

j_list = [4, 4, 4, 9, 10, 11, 12]
p_len = 4
random.shuffle(j_list)

def find_min_max_avg(j_list, p_len):
    j_list = sorted(j_list)
    if p_len <= len(j_list):
        min_avg = sum(j_list[:p_len])/p_len
        max_avg = sum(j_list[-p_len:])/p_len
        print(f'the min avg is: {min_avg}\nthe max avg is: {max_avg}')
    else:
        print('p_len is too large')

find_min_max_avg(j_list, 2)
find_min_max_avg(j_list, 3)
find_min_max_avg(j_list, 7)

from standard_py.my_module import *


import numpy as np

print(-5 ** 2)
print((-5) ** 2)
print(np.square(-5))

# ---- one edit away -----
# insert one char
# remove one char
# replace one char

import string




def one_edit_away(a, b):

    letters = list(string.ascii_lowercase)

    def insert_one(j, k):
        for l in letters:
            for idx in range(len(j)+1):
                j_plus_l = j[:idx] + l + j[idx:]
                if j_plus_l == k:
                    print('insert one')
                    return True
        return False

    def remove_one(j, k):
        for idx in range(len(j)):
            j_remove = j[:idx] + j[idx + 1:]
            if j_remove == k:
                print('remove one')
                return True
        return False

    def replace_one(j, k):
        for l in letters:
            for idx in range(len(j)):
                j_replace_l = j[:idx] + l + j[idx + 1:]
                if j_replace_l == k:
                    print('replace one')
                    return True
        return False

    return (insert_one(a, b) or insert_one(b, a)
            or remove_one(a, b) or remove_one(b, a)
            or replace_one(a, b) or replace_one(b, a))


one_edit_away('pea', 'peas')
one_edit_away('peas', 'pea')
one_edit_away('peet', 'pet')
one_edit_away('pttt', 'pet')

ex_lst = [('pea', 'peas'), ('pea', 'fleas'), ('pea', 'lea'), ('pea', 'seas'), ('peat', 'seat')]

for a, b in ex_lst:
    print(a, b)
    print(one_edit_away(a, b))


def my_add(a: int, b: int) -> int:
    return a + b

my_add(5, 6)
my_add('a', 6)


def fooie(n):
    '''Write a function that takes in an integer n, and prints out integers from 1 to n inclusive.
    If %3 == 0 then print "foo" in place of the integer,
    if %5 == 0 then print "ie" in place of the integer,
    and if both conditions are true then print "foo-ie" in place of the integer.
    '''
    for i in range(1, n+1):
        if i % 3 == 0 and i % 5 == 0:
            print('foo-ie')
        elif i % 3 == 0:
            print('foo')
        elif i % 5 == 0:
            print('ie')
        else:
            print(i)

    print('next')
    for i in range(1, n+1):
        res = ''
        if i % 3 == 0:
            res += 'foo'
        if i % 5 == 0:
            if res:
                res += '-'
            res += 'ie'
        if not res:
            res = str(i)
        print(res)


fooie(30)

import antigravity

import itertools as it

perms = list(it.permutations('ABCD', 3))
combs = list(it.combinations('ABCD', 3))

from pprint import pprint

pprint(perms)
pprint(combs)



if __name__ == '__main__':
    print('this will not run when the module is imported')

    a_dog = Dog('bud', 'lab')
    print(a_dog)
    run_ret = a_dog.run(10)
    print('run:', run_ret)
    print(a_dog.is_sit())
    a_dog.do_sit()
    print(a_dog.is_sit())
    a_dog.do_sit()
    print(a_dog.is_sit())




