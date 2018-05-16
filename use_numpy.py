# reference: numpy.pdf



import numpy as np
# The numpy (numeric python) package provides basic routines for 
# manipulating large arrays and matrices of numeric data.



# ---------------------------------------------------------------
#                               Arrays
# ---------------------------------------------------------------
# The central feature of NumPy is the array object class.
# Arrays are similar to lists in Python, except that every
# element of an array must be of the same type, typically
# a numeric type like float or int. Arrays make operations
# with large amounts of numeric data very fast and are
# generally much more efficient than lists.





# ---------------------------------------------------------------
# create arrays
# one dimensional array
a = np.array([1, 3, 6, 8], dtype=float)
a
type(a)


# multi-dimensional array
# A two-dimensional array is a matrix.
b = np.array([[1,2,3], [4,5,6]], dtype=float)
b



# ---------------------------------------------------------------
# access elements
# Array elements are accessed, sliced, and manipulated just
# like lists.
a[0]
a[1]
a[:2]


b[0, 0]
b[0:2, 0:2]
b[0, :]
b[:, 2]




# ---------------------------------------------------------------
# get properties of an array
# get size
a.shape
b.shape

# get data type
b.dtype
# float64 is a numeric type that numpy uses to store double-precision
# (8-byte) real numbers, similar to the float type in Python.

# get length
len(a)




# ---------------------------------------------------------------
# test if values are present in an array
2 in b
8 in b



# ---------------------------------------------------------------
# reshape
a = np.array(range(12), dtype=float)
b = a.reshape((3,4))
# notice that the reshape function creates a new array and does not
# modify hte original array.



# ---------------------------------------------------------------
# from array to list and vice versa
a = np.array([1,2,3], dtype=float)
b = a.tolist()
c = list(a)


c = np.array(b, dtype=int)




# ---------------------------------------------------------------
# transpose
a = np.array(range(12), dtype=float).reshape((6,2))
a
b = a.transpose()
b



# One-dimensional versions of multi-dimensional arrays can be
# generated with the flatten function
a = np.array([[1, 2, 3], [4, 5, 6]], float)
a
a.flatten()




# ---------------------------------------------------------------
# concatenation
a = np.array([[1,2], [3,4]], dtype=float)
b = np.array([[5,6], [7,8]], dtype=float)
c = np.concatenate((a,b), axis=0)
d = np.concatenate((a,b), axis=1)



# ---------------------------------------------------------------
# other ways to create arrays
np.arange(5, dtype=float)
# the arange function is similar to the range function but returns 
# an arrays
np.arange(1,10,3, dtype=int)



# special matrices
np.ones((3,4), dtype=float)
np.zeros((3,4), dtype=int)
np.identity(4, dtype=float)




# ---------------------------------------------------------------
# Array mathematics
# When standard mathematical operations are used with arrays, they
# are applied on an element-by-element basis. This means that the
# arrays should be the same size during addtion, subtraction, etc.
a = np.array([7,2,3], dtype=float)
b = np.array([5,2,8], dtype=float)

a + b
a - b
a * b
a / b
a % b
b ** a


# For two-dimensional arrays, multiplication remains elementwise and
# does not correspond to matrix multiplication. 
a = np.array([[1,2],[3,4]], dtype=float)
b = np.array([[2, 0], [1,3]], dtype=float)
a
b
a * b



# In addition to the standard operators, NumPy offers a large library
# of common mathematical functions that can be applied elementwise to
# arrays. Among these are the functions: abs, sign, sqrt, log, log10,
# exp, sin, cos, tan, etc.
a = np.array([1,4,9], dtype=float)
np.sqrt(a)


# The functions floor, ceil, and rint give the lower, upper, or nearest
# (rounded) integer.
a = np.array([1.1, 1.5, 1.9], dtype=float)
np.floor(a)
np.ceil(a)
np.rint(a)


# Also included in the NumPy module are two important mathematical constants.
np.pi
np.e 




# ---------------------------------------------------------------
# Array iterations

a = np.array([1,2,3], dtype=float)
for x in a:
    print x
    
    
    
    
# ---------------------------------------------------------------
# Basic array operations
  
a = np.array([2, 5, 9], dtype=float)

# Member functions of the arrays can be used.
a.sum()
a.prod()    


# Alternatively, standalone functions in the NumPy module can be used.
np.sum(a)
np.prod(a)


# statistical quantities
a.mean()
a.var()
a.std()
a.min()
a.max()
a.argmin()
a.argmax()



# For multidimensional arrays:
a = np.array([[0, 5, 4], [4, -1, 7], [2, 9, 1], [6, -3, 6]], dtype=float)
a
a.mean(axis=0)
a.mean(axis=1)




# Like lists, arrays can be sorted.
a = np.array([4, 1, -9, 4], dtype=float)
a.sort()


# Values in an array can be clipper to be within a prespecified range.
a = np.array([6, 2, 5, -1, 0], float)
a
a.clip(0, 5)


# unique elements
a = np.array([1, 1, 4, 5, 5, 5, 7], float)
a
np.unique(a)



# extract diagonal from a two-dimensional matrix
a = np.array([[1, 2], [3, 4]], float)
a
a.diagonal()



# ---------------------------------------------------------------
# comparison operators
a = np.array([1, 3, 0], float)
b = np.array([0, 3, 2], float)
a > b
a == b
a <= b
a > 2  


#The any and all operators can be used to determine whether or not any or 
#all elements of a Boolean array are true:
c = a > 2
any(c)
all(c)



# Compound Boolean expressions can be applied to arrays on an element-by-element 
# basis using special functions logical_and, logical_or, and logical_not.
a = np.array([1, 3, 0], float)
a
b = np.logical_and(a > 0, a < 3)
b
np.logical_not(b)
c = np.array([False, True, False], bool)
np.logical_or(b, c)



# ---------------------------------------------------------------
# random numbers
np.random.seed(293423)



# uniform distribution
# to generate an array of random numbers in the half-open interval [0.0, 1.0)
np.random.rand(5)
# to generate 2-D random arrays
np.random.rand(3,2)
np.random.rand(6).reshape((2,3))
# to generate a single random number in [0.0, 1.0)
np.random.random()
# to generate random integers in the range [min, max]:
np.random.randint(5, 10)





# binomial
n = 100
p = 0.3
size = 100
np.random.binomial(n=n, p=p, size=size)



# normal
np.random.normal(loc=1.5, scale=4.0, size=10)
# to draw from a standard normal distribution
np.random.normal(size=10)
