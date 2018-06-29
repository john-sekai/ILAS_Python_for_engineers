# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:48:52 2018

@author: rpf19
"""

import numpy as np
import matplotlib.pyplot as plt
import random 
import scipy

# numpy can solve linear equations
A = np.array([[1,-1,1],
              [0,10,25],
              [20,10,0]])
B = np.array([0,90,80]) # 3 x 1 vector
print("A = ",A)
print("B = ",B)
print("B with shape:",B.shape)
print()

x = np.linalg.solve(A,B)
print(x)

print()
print(np.dot(A,x))
# @ is of the same function as np.dot
print(A @ x)

# use matrix inverse to solve the equation
# use np.matmul instead of simple '*'
x = np.matmul(np.linalg.inv(A), B)
print(x)

xs = np.array([-2,1,4])
print("xs = ",xs)
xs = xs.reshape((3,1))
print("Now, xs =",xs)
print()

power = np.array([2,1,0])
x = np.power(xs,power)
print(x)

# choose a random sample
x = np.linspace(-6,10,100)
# using np.random.random to return random float between (0.0,1.0)
y = (2*x**3 - 8*x**2 - 96*x) * np.random.random(size=x.shape)
# zip 为打包成元组，长度和最小的元素一致
x_sub, y_sub = zip(*random.sample(list(zip(x,y)),50))

a = np.polyfit(x_sub,y_sub,3)
yfit3 = a[0]*x**3 + a[1]*x**2 + a[2]*x + a[3]
r = np.roots(a)
z = np.zeros(len(r))
plt.plot(x,yfit3,'r--')
plt.plot(r,z,'ks',markersize=10)

# counting roots
print("length of root:\n",len(r))
print()

print("length of root:\n",np.sum(y[:-1] * y[1:] < 0))
#print("WHY WORKING:\n",y[:-1] * y[1:] < 0)
plt.show()

# iterative solutions
m = 1.0
L = m**3 / 1000.0
mol = 1.0
s = 1.0
Cao = 2.0*mol/L
V = 10.0*L
k = 16

c = np.linspace(0.001,2) * mol / L

def func(Ca):
    return V -(Cao - Ca) / (k * Ca**2)
plt.plot(c,func(c))
plt.xlabel('Ca (mol/m^3)')
plt.xlabel('f{Ca}')
plt.ylim([0,0.012])
plt.show()
# run iterative solver on the input data
# fsolve(func,root_guess) returns the closest root of func = 0
# to root_guess
# the tolerance of root xtol default to 1.49012e-08
from scipy.optimize import fsolve

cguess = 125
c_root = fsolve(func,cguess)
print(c_root)

# bisection
from scipy.optimize import bisect # bisection method
from scipy.optimize import brentq # Classic Brent method 

def F(x):
    return (4*x**3 -3*x**2 -25*x -6)
x = np.linspace(-4,5)
y = F(x)
plt.plot(x,y)
plt.grid(color='k',linestyle='-', linewidth=0.5)
plt.show()

print("y = ",y)
print("y[:-1] = ",y[:-1])
print("y[1:] = ",y[1:])
i = np.argwhere(y[:-1]*y[1:] < 0)
int_low = x[:-1][i]
int_hi = x[1:][i]

root = scipy.optimize.bisect(F, int_low[2][0], int_hi[2][0])
print(root)
root = scipy.optimize.brentq(F, int_low[2][0], int_hi[2][0])
print(root)

# finding the n-th root of a periodic function
# example find the root of the cosine function in vicnity of -5
from numpy import pi as pi

root = scipy.optimize.bisect(np.cos, -5,-2.5)
print(root/pi,'pi')
root = scipy.optimize.brentq(np.cos, -5,-2.5)
print(root/pi,'pi')
print('\n\n')
# solving coupled non-linear equations
def objective(guess):
    x,y = guess
    z1 = y - x**2
    z2 = y-8 + x**2
    return [z1,z2]
guess = [1,1]
sol = fsolve(objective,guess)
print(sol)