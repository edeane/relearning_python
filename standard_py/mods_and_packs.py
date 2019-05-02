"""
Python Modules and Packages - An Introduction
https://realpython.com/python-modules-packages/

https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html

"""


import sys
from pprint import pprint
from importlib import reload

pprint(sys.path)

# sys.path.append(r'C:Users/edeane/PycharmProjects')

import pkg
from pkg import mod1
from pkg import mod2

reload(mod1)

pkg.a_list

mod1.greeting('eddie')
mod1.get_id(123)
mod1.get_id(111)
mod1.fact(40)


mod2.foo()
mod2.bar()


