"""
Python itertools
- https://realpython.com/python-itertools/
- https://pymotw.com/3/itertools/index.html
- https://docs.python.org/3/library/itertools.html
- https://docs.python.org/3.7/library/itertools.html#itertools-recipes

iterator algebra
building blocks that can be combined to form specialized data pipeline

"""

import itertools as it
import operator as op
import random

list(zip([1,2,3],['a','b','c']))

iter([1,2,3,4])

list(map(len, ['abc', 'de', 'fghi']))

list(map(sum, zip([1,2,3], [4,5,6])))



inputs = list(range(1, 11))
n = 2
# given list of inputs and n
# break list into groups of n


[tuple(inputs[i*n:(i+1)*n]) for i in range(len(inputs) // n)]

iters = [iter(inputs)] * n
list(zip(*iters))

# same id
[id(itr) for itr in iters]
[list(itr) for itr in iters]

x = list(range(1, 6))
y = ['a', 'b', 'c']

len(x)
len(y)

list(zip(x, y))
list(it.zip_longest(x, y))

def a_grouper(inputs, n, fillvalue=None):
    iters = [iter(inputs)] * n
    return it.zip_longest(*iters, fillvalue=fillvalue)


list(a_grouper(list(range(1, 11)), 4))


# You have
# Three $20 dollar bills
# Five $10 dollar bills
# Two $5 dollar bills
# Five $1 dollar bills
# How many ways can you make change for a $100 dollar bill?
# Choice of k things from n things is a combo

wallet = ([20] * 3) + ([10] * 5) + ([5] * 2) + ([1] * 5)
wallet
len(wallet)

len(list(it.combinations(wallet, 3)))

makes_100 = []
for n in range(1, len(wallet)+1):
    for combo in it.combinations(wallet, n):
        if sum(combo) == 100:
            makes_100.append(combo)

len(makes_100)
set(makes_100)




list(it.combinations_with_replacement([1,2], 2)) # [(1, 1), (1, 2), (2, 2)]
list(it.combinations([1,2], 2)) # [(1, 2)]
# order matters for permutations
list(it.permutations([1,2], 2)) # [(1, 2), (2, 1)]

# How many ways are there to make change for a $100 bill using any number of $50, $20, $10, $5, and $1 dollar bills?
bills = [50, 20, 10, 5, 1]
makes_100 = []
for n in range(1, 101):
    for combo in it.combinations_with_replacement(bills, n):
        if sum(combo) == 100:
            makes_100.append(combo)

len(makes_100)


a_list = [1, 2, 3]

list(it.combinations(a_list, 2))
# [(1, 2), (1, 3), (2, 3)]

list(it.combinations_with_replacement(a_list, 2))
# [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

list(it.permutations(a_list, 2))
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]


def evens():
    """Generate even integers, starting with 0"""
    n = 0
    while True:
        yield n
        n += 2

def odds():
    """Generate even integers, starting with 0"""
    n = 1
    while True:
        yield n
        n += 2


evens_n = evens()
type(evens_n)
[next(evens_n) for _ in range(6)]


counter = it.count(step=2)
type(counter)
[next(counter) for _ in range(6)]

odds_n = it.count(start=1, step=2)
[next(odds_n) for _ in range(3)]

# is.count creates an infinite iterator

list(zip(it.count(start=1, step=2), ['a', 'b', 'c']))


# good ole fibs

def fibs_ret(n):
    a, b = 0, 1
    result = []
    for i in range(n):
        result.append(a)
        a, b = b, a + b
    return result

fibs_ret(10)


def fibs():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

a_fib = fibs()
[next(a_fib) for _ in range(10)]


list(zip(it.repeat(5), ['a', 'b', 'c']))

all_ones = it.repeat(1)

[next(all_ones) for _ in range(5)]
list(it.repeat(2, 5))

alt_ones = it.cycle([1, -1])
[next(alt_ones) for _ in range(10)]




list(it.accumulate([1,2,3,4,5], op.add))

# running minimum

b_list = [9, 21, 17, 5, 11, 12, 2, 6]

list(it.accumulate(b_list, min))

a_list = list(range(1, 6))
list(it.accumulate(a_list, lambda x, y: (x+y)/2))

# accumulate is the previous result


def accum_fun(inputs, func):
    itr = iter(inputs)
    prev = next(itr)
    for cur in itr:
        yield prev
        prev = func(prev, cur)
    yield prev


accum_add = accum_fun([1,2,3,4,5], op.add)
list(accum_add)


list(it.accumulate(range(1,6), op.sub))
list(it.accumulate(range(1,6), lambda x, y: x - y))

# Building a Poker app




deck_one = cards()
list(deck_one)

deck_two = ((rank, suit) for rank in ranks for suit in suits)

# Cartesian product

deck = list(it.product(ranks, suits))
list(it.product([1,2,3], ['a', 'b'], ['c', 'd']))



for i in iter('abcdefg'):
    print(i)

a_string = 'abcdefghijklmnopqrstuvwxyz'
a_string_iter = iter(a_string)
next(a_string_iter)



iter1, iter2, iter3 = it.tee([1,2,3,4,5], 3)
list(iter1)
list(iter1)
list(iter2)


# slice iterable
list(it.islice(range(10), 5))
list(it.islice(range(10), 0, 5))
list(it.islice(range(10), 5, None))



# Poker App

ranks = ['A', 'K', 'Q', 'J'] + list(map(str, reversed(range(2, 11))))
suits = ['H', 'D', 'C', 'S']

def cards():
    """Return a generator that yields playing cards."""
    for rank in ranks:
        for suit in suits:
            yield rank, suit

def shuffle(deck):
    """Return iterator over shuffled deck."""
    deck = list(deck) # from iter to object in memory
    random.shuffle(deck) # fisher-yates shuffle
    return iter(tuple(deck))

def cut(deck, n):
    """Return an iterator over a deck of cards cut at index 'n'."""
    if n < 0:
        raise ValueError("'n' must be a non-negative integer")

    # deck = list(deck)
    # return iter(deck[n:] + deck[:n])

    deck1, deck2 = it.tee(deck, 2)
    top = it.islice(deck1, n, None)
    bottom = it.islice(deck2, 0, n)
    return it.chain(top, bottom)



def deal_one(deck, num_hands=2, hand_size=5):
    result = [[] for _ in range(num_hands)]
    for n in range(hand_size):
        for num_hand in range(num_hands):
             result[num_hand].append(next(deck))
    return [tuple(i) for i in result]

def deal_two(deck, num_hands=2, hand_size=5):
    result = []
    for i in range(hand_size):
        dealts = []
        for j in range(num_hands):
            dealts.append(next(deck))
        result.append(tuple(dealts))
    result = tuple(result)
    return list(zip(*result))

def deal_crazy(deck, num_hands=1, hand_size=5):
    # creates a list of iter decks referencing the same memory id
    iters = [iter(deck)] * hand_size
    return list(zip(*(tuple(it.islice(itr, num_hands)) for itr in iters)))



deck = cards()

list(deal_one(deck, 2, 4))
list(deal_two(deck, 2, 4))
list(deal_crazy(deck, 2, 4))

deck = shuffle(deck)
deck = cut(deck, 1)
list(deck)

list(it.chain([1,2,3],
              [4,5,6]))

list(it.chain.from_iterable([
    [1, 2, 3],
    [4, 5, 6],
]))


cycle = it.chain.from_iterable(it.repeat('abc'))
list(it.islice(cycle, 10))


# Analyze the S&P500 with itertools
# Determine the maximum daily gain, daily loss (in percent change), and the longest growth streak


from collections import namedtuple
import csv
from datetime import datetime
from pprint import pprint
import functools as ft
import itertools as it


class DataPoint(namedtuple('DataPoint', ['date', 'value'])):
    __slots__ = ()

    def __le__(self, other):
        return self.value <= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value


def read_prices(csvfile, _strptime=datetime.strptime):
    with open(csvfile) as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            yield DataPoint(date=_strptime(row['Date'], '%Y-%m-%d').date(),
                            value=float(row['Adj Close']))

prices = tuple(read_prices('data/SP500.csv'))
len(prices)
pprint(prices[:10])
pprint(prices[-10:])


# Determine the maximum daily gain

tuple(DataPoint(day.date, 100*(day.value/prev_day.value - 1.)) for day, prev_day in zip(prices[1:], prices))

gains = []
for day, prev_day in zip(prices[1:], prices):
    gains.append(DataPoint(day.date, 100*(day.value/prev_day.value - 1)))

gains = tuple(gains)
pprint(gains[:10])
pprint(prices[:10])

max_gain = DataPoint(date=None, value=0)
for data_point in gains:
    max_gain = max(data_point, max_gain)

print(max_gain)


zdp = DataPoint(None, 0)
print(ft.reduce(max, gains, zdp))
print(ft.reduce(min, gains, zdp))


import numpy as np
import pandas as pd

sp_df = pd.read_csv('data/SP500.csv')

sp_df['gains'] = 100 * sp_df['Adj Close'].pct_change()
sp_df.head()

sp_df.loc[sp_df['gains'] == sp_df['gains'].max(), ['Date', 'gains']]
sp_df.loc[sp_df['gains'] == sp_df['gains'].min(), ['Date', 'gains']]
sp_df['gains_sign'] = sp_df['gains'].apply(np.sign)
sp_df['streak'] = sp_df['gains_sign'].groupby((sp_df['gains_sign'] != sp_df['gains_sign'].shift()).cumsum()).cumsum()


def get_streak(df, col_name='streak', max_or_min='max'):
    """Calc streak"""
    if max_or_min not in ['max', 'min']:
        raise ValueError("max_or_min must be either 'max'' or 'min'")

    if max_or_min == 'max':
        n = df[col_name].max()
    elif max_or_min == 'min':
        n = df[col_name].min()

    idx_to = df[df[col_name] == n].index[0] + 1
    idx_from = int(idx_to - abs(n))
    from_date = sp_df.loc[idx_from, 'Date']
    to_date = sp_df.loc[idx_to, 'Date']
    print(f'From: {from_date} To: {to_date} {max_or_min.title()} Streak: {int(n)}')


get_streak(sp_df, max_or_min='max')
get_streak(sp_df, max_or_min='min')











