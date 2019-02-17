# -*- coding: utf-8 -*-
"""Fibonacci numbers module

This module has functions for the fibonacci sequence.

Example:
    $ python fibo.py

Todo:
    * Nothing 1
    * Nothing as well 2
    * and nothing 3

"""


fib_name = 'Fibonacci'
a_integer = 42
a_list = [x**2 for x in range(100)]
a_dict = {x: x**3 for x in range(0, 200, 2)}

print('Welcome to the fib module\n the name of this file is {0}'.format(__name__))
print('-' * 40)


def fib(n):
    """Prints Fibonacci series up to n

    Args:
        n (int): n integers to loop and print out
    Returns:
        None
    """
    a, b = 0, 1
    while a < n:
        print(a, end=', ')
        a, b = b, a+b
    print()


def fib2(n):
    """Returns Fibonacci series up to n

    Args:
        n (int): n integers to loop and add to list
    Returns:
        List with Fibonacci series
    """
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result


if __name__ == '__main__':
    print('the __name__ of the file is: {0}'.format(__name__))
    import sys
    fib(int(sys.argv[1]))
