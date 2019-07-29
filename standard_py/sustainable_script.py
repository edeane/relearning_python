"""Simple fizzbuzz generator.

This script prints out a sequence of numbers from a provided range with the following restrictions:
- if the number is divisible by 3, then print "fizz"
- if the number is divisible by 5, then print "buzz"
- if the number is divisible by 3 and 5, then print "fizzbuzz"


Sustainable Script
https://vincent.bernat.ch/en/blog/2019-sustainable-python-script

"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(os.path.splitext(os.path.basename(sys.argv[0]))[0])

def parse_args():
    """Parse arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=int, help='Start value')
    parser.add_argument('end', type=int, help='End value')

    parser.add_argument("--debug", "-d", action="store_true", default=False, help="enable debugging")
    parser.add_argument("--silent", "-s", action="store_true", default=False, help="don't log to console")

    g = parser.add_argument_group('fizzbuzz settings')
    g.add_argument('--fizz', metavar='N', default=3, type=int, help="Modulo value for fizz")
    g.add_argument('--buzz', metavar='N', default=5, type=int, help="Modulo value for buzz")

    return parser.parse_args()

def setup_logging(options):
    pass


def fizzbuzz(fizz, buzz):
    if n % fizz == 0 and n % buzz == 0:
        return 'fizzbuzz'
    elif n % fizz == 0:
        return 'fizz'
    elif n % buzz == 0:
        return 'buzz'
    else:
        return n


def run_fb(start, end, fizz, buzz):
    res = []
    for n in range(start, end):
        res.append(fizzbuzz(fizz, buzz))




if __name__ == '__main__':

    options = parse_args()
    fizzbuzz(options.start, options.end, options.fizz, options.buzz)


