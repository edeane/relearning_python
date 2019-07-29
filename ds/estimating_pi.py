"""

Estimating Pi
http://www.codedrome.com/estimating-pi-in-python/
https://vincent.bernat.ch/en/blog/2019-sustainable-python-script
https://towardsdatascience.com/hands-on-bayesian-statistics-with-python-pymc3-arviz-499db9a59501
"""

import math
from fractions import Fraction
from colorama import Fore, Style
from standard_py.learning_colorama import print_color

PI_STRING = f'{math.pi:1.48f}'


def print_as_text(pi):
    """
    Takes a value for pi and prints it below a definitive value,
    with matching digits in green and non-matching digits in red
    """

    pi_string = f'{pi:1.48f}'
    print("Definitive: " + PI_STRING)

    res = ''
    for i in range(0, len(pi_string)):
        if pi_string[i] == PI_STRING[i]:
            res = res + Fore.GREEN +  pi_string[i]
        else:
            res = res + Fore.RED + pi_string[i]

    res = res + Style.RESET_ALL
    print(f"Estimated:  {res}")
    return None

# Fractions

fractions_lst = [22/7, 333/106, 355/113, 52163/16604, 103993/33102, 3126535/995207, 245850922/78256779]

for fraction in fractions_lst:
    print(str(Fraction(fraction).limit_denominator(max_denominator=100_000_000)))
    print_as_text(fraction)


# Francois Viete

n_iter = 1000
num = 0
pi = 1

for i in range(n_iter):
    num = (2 + num) ** .5
    pi *= (num / 2)

print_as_text(2 / pi)



# John Wallis

n_iter = 100_000_000
num = 0
den = 1
pi = 1

for i in range(n_iter):
    if i % 2 == 0:
        num += 2
    else:
        den += 2
    pi *= (num/den)


print_as_text(pi * 2)

# John Machin

pi = 4 * (4 * math.atan(1/5) - math.atan(1/239))
print_as_text(pi)


# Gregory-Leibniz
n_iter = 400_000
den = 1
pi = 0
mult = -1

for i in range(n_iter):
    mult *= -1
    pi += mult * (4 / den)
    den += 2

print_as_text(pi)


# Nilakantha

n_iter = 1_000_000
pi = 3
mult = -1

for i in range(n_iter):
    mult *= -1
    ii = i * 2
    pi += mult * (4 / ((ii+2) * (ii+3) * (ii+4)))

print_as_text(pi)








