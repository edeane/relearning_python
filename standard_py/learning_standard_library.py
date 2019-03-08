# -*- coding: utf-8 -*-
"""
Brief Tour of the Standard Library - Part I and II
https://docs.python.org/3/tutorial/stdlib.html
https://docs.python.org/3/tutorial/stdlib2.html
"""

# ---- Part I ----

import os
import shutil
import glob
import sys
import re
import math
import random
import statistics
from urllib.request import urlopen
from datetime import date
import timeit
import doctest
import numpy as np


the_current_working_directory = os.getcwd()
print('the current working directory is', the_current_working_directory)
os_module_functions = dir(os)
os_module_functions
for i, os_mod_fun in enumerate(os_module_functions):
    print('index:', i, 'fun:', os_mod_fun )

shutil.copyfile('requirements.txt', r'C:\Users\edeane\\Documents\realearing_py_req.txt')

all_py_files = glob.glob('*py')
print(all_py_files, sep='*')

for i, a in enumerate(sys.argv):
    print(i, a)
print(sys.argv)


sys.stderr.write('Warning, log file not found starting a new one\n')

for i in range(10):
    sys.stdout.write('Hello stdout write ' + str(i) + '\n')


# regular expresions
re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest')
tea_string = 'tea for too'
tea_string = tea_string.replace('too', 'two')
print(tea_string)

# random
random.choice(['apple', 'pear', 'banana'])
random.sample(range(100), 10) # without replacement
random.random()
random.randrange(6)


# math

math.cos(math.pi / 4)
math.log(1024, 2)

# statistics

data = [2.75, 1.75, 1.25, .25, .5, 1.25, 3.5]
statistics.mean(data)
statistics.median(data)
statistics.variance(data)
statistics.stdev(data)
statistics.variance(data)**.5


with urlopen('http://tycho.usno.navy.mil/cgi-bin/timer.pl') as response:
    for line in response:
        line = line.decode('utf-8')
        print(line)


now = date.today()
now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B.")

s = '''\
res = []
for i in range(10000):
    res.append(i*2)
'''

timeit.timeit(s, number=100)

list(map(str, range(100)))

#doctest
# there is allso unittest which is more comprehensive

def average(values):
    """Returns the average of the input values

    Args:
        values: any iterable

    Returns:
        Average value

    Examples:
        Here is an example of it's usage:

    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)


data = [1,2,3,4,5,6,7,8,9,10]
average(data)
doctest.testmod()


# operator
# functools
# map
# reduce
# filter

# ---- Part II ----


import reprlib
import pprint
import textwrap
import locale
from string import Template
import time
import os
import struct
import threading, zipfile
import logging
import weakref, gc
from array import array
from collections import deque
import bisect
from heapq import heapify, heappop, heappush
from decimal import Decimal, getcontext
import json

# reprlib
long_word = 'supercalifragilisticexpialidocious'
repr(set(long_word))
reprlib.repr(set(long_word))


obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
              {"name": "Katie", "age": 38,
               "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""

json_obj = json.loads(obj)

reprlib.repr(json_obj)

from pprint import pprint
# pprint

t = [[[['black', 'cyan'], 'white', ['green', 'red']], [['magenta', 'yellow'], 'blue']]]
print(t)
pprint(t, width=60)
pprint(json_obj)

# textwrap
doc = """The wrap() method is just like fill() except that it returns
a list of strings instead of one big string with newlines to separate
the wrapped lines."""
print(textwrap.fill(doc, width=40))


# locale

locale.setlocale(locale.LC_ALL, '')
pprint(locale.localeconv())


# threading


# logging

logging.debug('debugging info')
logging.info('informational message')
config_file_name = 'configaroooo'
logging.warning(f'warning config file {config_file_name} not found')
logging.error('a big fat error occurred')
logging.critical('critical error -- shutting down')


# arrays

a = array('H', [4000, 10, 700, 22222])
sum(a)
sum([4000, 10, 700, 22222])


# deque

d = deque(['task1', 'task2', 'task3'])
d.append('task4')
print('handling', d.popleft())


# bisect

scores = [(100, 'perl'), (200, 'tcl'), (400, 'lua'), (500, 'python')]

bisect.insort(scores, (300, 'ruby'))
scores

# heapq
data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
heapify(data)
data
heappush(data, -5)
data
[heappop(data) for i in range(3)]
heappop(data)


round(Decimal('0.70') * Decimal('1.05'), 2)

round(.70 * 1.05, 2)


round(.7 * 1.05, 2)

for i in range(1, 15, 2):
    n = i / 2
    # print(n, '===>', round(n), np.round(n), Decimal(str(n)).quantize(0, ROUND_HALF_UP))

from decimal import ROUND_HALF_UP
ROUND_HALF_UP

Decimal('2.5').quantize(0, ROUND_HALF_UP)


round(1.555, 2)
round(1.556, 2)
round(1.554, 2)
round(1.155, 2)
round(1.255, 2)
round(1.355, 2)
round(1.455, 2)


Decimal(0.1)


sum([Decimal('0.1')] * 10) == Decimal('1.0')


sum([0.1] * 10) == 1.0

(0.1 * 10) == 1.0

1 % .1

10 % 4
10 % 3







