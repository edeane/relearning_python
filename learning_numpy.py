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

# starting with https://docs.scipy.org/doc/numpy/user/quickstart.html
a = np.arange(15).reshape(3,5)
print(a.shape)
a.ndim
a.dtype.name
a.size
a.shape[0] * a.shape[1]
type(a)
b = np.array([6, 7, 8])
b
type(b)

c = np.array([ [1,2], [3,4] ], dtype=complex)
c

np.zeros( (3,4) )
np.ones( (2,3,4) , dtype=np.int16)

np.empty( (2,3) )

np.arange(10, 30, 5)
np.linspace(0, 2, 9)

x = np.linspace(0, 2*np.pi, 100)
f = np.sin(x)
f

print(np.arange(10000).reshape(100,100))

a = np.array( [20,30,40,50] )
b = np.arange(4)
a-b


b
b**2
10*np.sin(a)
a < 35



a = np.array( [[1,1],
               [0,1]])

b = np.array( [[2,0],
               [3,4]])

#element wise
a * b

# matrix product
a @ b
a.dot(b)

# inplace
a = np.ones((2,3),dtype=int)
a *= 3
a

b = np.random.random((2,3))
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
d = np.exp(c*1j)
d

1j
a = np.array([1,2,3,4,5])
np.exp(a)

import math

math.e ** 1
math.e ** 2


a = np.random.random((2,3))
a.sum()
a.min()
a.max()


# axis

b = np.arange(12).reshape(3, 4)
b.sum(axis=0) # sum each column
b.sum(axis=1) # sum each row
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


b = np.fromfunction(f, (5,4), dtype=int)
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




a = np.floor(10*np.random.random((2,2)))
a
b = np.floor(10*np.random.random((2,2)))
b
np.vstack((a,b))
np.hstack((a,b))

# newaxis

a = np.linspace(2, 10, 5)
a
a = np.arange(2, 12, 2)
a
a[:, np.newaxis]

from pprint import pprint

a = np.floor(10*np.random.random((2,12)))

pprint(np.hsplit(a, 3))

pprint(np.hsplit(a, (3,4))) # after 3rd and 4th column


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

a.argmax() # index of max
a.argmin() # index of min
a.argsort()



# complex slicing


# left off here https://docs.scipy.org/doc/numpy/user/quickstart.html#fancy-indexing-and-index-tricks










