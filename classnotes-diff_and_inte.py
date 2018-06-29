# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:36:10 2018

@author: rpf19
"""

import numpy as np
import scipy
import sympy

import matplotlib.pyplot as plt

# numerical derivatives



# numpy method 
p = np.poly1d([1,1,1,1])
pd = np.polyder(p)
print(pd)
# find the derivative at x = 2
print(pd(2.0))

# use polyder to get the arbitrary order derivative
print(np.polyder(p,2)) # second order derivative

# for piecewise function
def f1(x): 
    if x < 0:
        return 0
    elif 0 <= x <= 1:
        return x 
    elif 1 < x <= 2:
        return 2.0 - x 
    else:
         return x + 1.0
print(f1(-1))
#print(f1([0,1,2,3,4]))
f1 = np.vectorize(f1)
print(f1([0,1,2,3,4]))
print()

# integration
from scipy.integrate import quad
def integrand(x):
    return x**2

ans,err = quad(integrand,0,100)
print(ans)
print(f'The estimated error is {err}')
print()
from scipy.integrate import dblquad,tplquad,nquad
a,b,c,d = 1, 10, 0, 0.5
f = lambda x,y:1
area = dblquad(f,c,d,
               lambda x: a,
               lambda x: b)
print(area)

a,b,c,d,e,f = 0,1,2,3,1,2

f = lambda z,y,x,k: k*x*y*z

# Arguments:
# function, outer integral limits --> inner integral limits
a = tplquad(f,1,2,
            lambda x:2, lambda x:3,
            lambda x,y :0, lambda x, y: 1,
            args=(3,))
print(a)
print()
a = nquad(nquad(f,[[0,1],
                   [2,3],
                   [1,2]],
                    args=(3,)))






