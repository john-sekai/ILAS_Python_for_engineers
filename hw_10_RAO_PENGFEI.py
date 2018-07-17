# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 23:02:54 2018

@author: rpf19
"""

import numpy as np
import matplotlib.pyplot as plt

# Example 1 --- move the definition of xvalues to the first line
xvalues = np.linspace(-3, 3, 100)
y = (xvalues + 2) * (xvalues - 1) * (xvalues - 2)
plt.plot(xvalues, y, 'r--')
plt.plot([-2, 1, 2], [0 ,0, 0], 'bo', markersize=10)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Nice Python figure!')
plt.show()

# Example 2 -- modify the calling of the test function and the linspace
def test(x, alpha):
    return np.exp(-alpha * x) * np.cos(x)
x = np.linspace(0, 10*np.pi, 100)
alpha = 0.2
y = test(x,alpha)
plt.plot(x, y, 'b')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Example 3 -- modify the for and the print
a = np.array([2, 2, 4, 2, 4, 4])

for i in range(len(a)):
    if a[i] < 3:  # replace value with 77 when value equals 2
        a[i] = 77
    else:  # otherwise replace value with -77
        a[i] = -77
print('modified a:', a)

# Example 4 -- size of zeros should be in cell
y = np.zeros((20, 20))
y[8:13] = 10
plt.matshow(y)
plt.title('image of array y');
plt.show()

def gravity(M,m,r,G = 6.674e-11):
    if (G<0)or(m<0)or(M<0)or(r<0):
        raise ValueError('All input parameters should be non-negative.')
    else:
        return (G*M*m/r)
print()
try:
    force = gravity(10,10,-1,6.674e-11)
except ValueError:
    print('ValueError caught.')
        