"""
numpy
https://docs.scipy.org/doc/numpy/user/quickstart.html
http://cs231n.github.io/python-numpy-tutorial/
https://scipy-cookbook.readthedocs.io/items/idx_numpy.html

scipy
https://scipy-cookbook.readthedocs.io/items/FortranIO.html

pip install numpy scipy matplotlib ipython jupyter pandas sympy nose


Array Creation
    arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid,
    ogrid, ones, ones_like, r, zeros, zeros_like
Conversions
    ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat
Manipulations
    array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis,
    ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack
Questions
    all, any, nonzero, where
Ordering
    argmax, argmin, argsort, max, min, ptp, searchsorted, sort
Operations
    choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum
Basic Statistics
    cov, mean, std, var
Basic Linear Algebra
    cross, dot, outer, linalg.svd, vdot

"""

import numpy as np
import matplotlib.pyplot as plt

# starting with https://docs.scipy.org/doc/numpy/user/quickstart.html
a = np.arange(15).reshape(3, 5)
print(a.shape)
a.ndim
a.dtype.name
a.size
a.shape[0] * a.shape[1]
type(a)
b = np.array([6, 7, 8])
b
type(b)

c = np.array([[1, 2], [3, 4]], dtype=complex)
c

np.zeros((3, 4))
np.ones((2, 3, 4), dtype=np.int16)

np.empty((2, 3))

np.arange(10, 30, 5)
np.linspace(0, 2, 9)

x = np.linspace(0, 2 * np.pi, 100)
f = np.sin(x)
f

print(np.arange(10000).reshape(100, 100))

a = np.array([20, 30, 40, 50])
b = np.arange(4)
a - b

b
b ** 2
10 * np.sin(a)
a < 35

a = np.array([[1, 1],
              [0, 1]])

b = np.array([[2, 0],
              [3, 4]])

# element wise
a * b

# matrix product
a @ b
a.dot(b)

# inplace
a = np.ones((2, 3), dtype=int)
a *= 3
a

b = np.random.random((2, 3))
b += a
b

a += b
b.dtype
a = a.astype(float)
a += b
a

a = np.arange(11)

a.tobytes()

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, np.pi, 3)
a
b
c = a + b
c
d = np.exp(c * 1j)
d

1j
a = np.array([1, 2, 3, 4, 5])
np.exp(a)

import math

math.e ** 1
math.e ** 2

a = np.random.random((2, 3))
a.sum()
a.min()
a.max()

# axis

b = np.arange(12).reshape(3, 4)
b.sum(axis=0)  # sum each column
b.sum(axis=1)  # sum each row
b.min(axis=1)
b.min(axis=0)
b.cumsum(axis=1)

b = np.arange(3)
np.sqrt(b)
c
b
np.add(b, c)

a = np.arange(10) ** 3
a
a[2]
a[2:5]
a[::-1]


# row = x
# column = y
# row 2 column 3 = 23 with 0 indexing
def f(x, y):
    return (10 * x) + y


b = np.fromfunction(f, (5, 4), dtype=int)
b[2, 3]
b[0:5, 1]
b[0:5, 0]
b[1:3, ]
b[1:3]
b[:, 1]

b[-1, :]

np.array(b.flat)

a = np.floor(10 * np.random.random((3, 4)))
a.shape

np.array(a.flat)
a.ravel()

a.reshape(6, 2)
a
a.T

a = np.floor(10 * np.random.random((2, 2)))
a
b = np.floor(10 * np.random.random((2, 2)))
b
np.vstack((a, b))
np.hstack((a, b))

# newaxis

a = np.linspace(2, 10, 5)
a
a = np.arange(2, 12, 2)
a
a[:, np.newaxis]

from pprint import pprint

a = np.floor(10 * np.random.random((2, 12)))

pprint(np.hsplit(a, 3))

pprint(np.hsplit(a, (3, 4)))  # after 3rd and 4th column

a
id(a)

b = a
id(b)
id(a) == id(b)

b = a[:]
id(a) == id(b)

b = a.copy()
id(a) == id(b)
a
b

a.argmax()  # index of max
a.argmin()  # index of min
a.argsort()

# complex slicing


# left off here https://docs.scipy.org/doc/numpy/user/quickstart.html#fancy-indexing-and-index-tricks


a = np.arange(1, 5.5, .5)

a.cumsum()

a = np.arange(12) ** 2
a

idx = np.array([1, 1, 3, 8, 5])

a[idx]

j = np.array([[3, 4], [9, 7]])

a[j]

palette = np.array([[0, 0, 0],  # black
                    [255, 0, 0],  # red
                    [0, 255, 0],  # green
                    [0, 0, 255],  # blue
                    [255, 255, 255]])  # white

palette

image = np.array([[0, 1, 2, 0],
                  [0, 3, 4, 0]])

palette[image]

time = np.linspace(20, 145, 5)
data = np.sin(np.arange(20)).reshape(5, 4)
time
data

# 0 = down
# 1 = across

ind = data.argmax(axis=0)
ind

time[ind]

data[ind, range(4)]

a = np.arange(5)
a[[1, 2, 3]] = 0
a

a = np.arange(12).reshape(3, 4)
a

b = a > 4
a[b] = 0
a

maxit = 20
maxit

def mandelbrot(h, w, maxit=20):
    """Returns an image of the Mandelbrot fractal of size (h,w).

    Args:
        h: height
        w: widht
        maxit: max iterations
    Returns: matpotlib image
    """
    y, x = np.ogrid[-1.4:1.4:h * 1j, -2:0.8:w * 1j]
    c = x + y * 1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z ** 2 + c
        diverge = z * np.conj(z) > 2**2
        div_now = diverge & (divtime == maxit)
        divtime[div_now] = i
        z[diverge] = 2

    return divtime

mand_4_4 = mandelbrot(400, 400)
mand_4_4.shape

plt.imshow(mand_4_4)


a = np.arange(12).reshape(3,4)
b1 = np.array([False, True, True])
b2 = ([True, False, True, False])

a
a[b1, :]
a[:, b2]
a[b1, b2]

# Linear Algebra

a = np.arange(1, 5, dtype=float).reshape(2,2)
a
a.T
a.transpose()
np.linalg.inv(a)
u = np.eye(2)
u
j = np.array([[0, -1], [1,0]], dtype=float)
j
j @ j
np.trace(u)


a = np.arange(30)
a.reshape(2,5,3)
a.shape = (2,5,3)
a


x = {'a': 1, 'b': 2}
y = {'b': 3, 'c': 4}

z = {**x, **y}
z


# ---- cs231 ----



a_arr = [3, 8, 6, 5, 1, 2, 1]

pivot = a_arr[len(a_arr) // 2]
left = [ x for x in a_arr if x < pivot]
middle = [x for x in a_arr if x == pivot]
right = [x for x in a_arr if x > pivot]

print(pivot, left, middle, right)

len(a_arr) / 2




np.full((2,3), 7)



a = np.arange(1, 7).reshape(3, 2)
a[a>2]


a = np.arange(1, 5, dtype=np.float64).reshape(2, 2)
b = np.arange(5, 9, dtype=np.float64).reshape(2, 2)
a
b
a + b
a / b
a ** .5
np.sqrt(a)














