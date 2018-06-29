# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 13:58:29 2018

@author: rpf19
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as pi
from scipy.optimize import fsolve
from scipy.optimize import bisect
from scipy.optimize import brentq

t = np.array([0,0.25,0.5,0.75])# 4x1 vector
#print("shape of t:",t.shape)
def objective(t):
    # generate the coefficients of the linear equations
    return np.transpose(np.array([np.cos(pi*t), np.cos(2*pi*t),np.cos(3*pi*t),np.cos(4*pi*t)]))
def predict(coeff,t):
    # using the coefficients and time input to predic the output
    # coeff--4x1 vector
    # t-- nx1 time series
    temp = np.array([np.cos(pi*t), np.cos(2*pi*t),np.cos(3*pi*t),np.cos(4*pi*t)])
    return np.matmul(np.transpose(coeff),temp)
A_gen =objective(t)
#A = np.array([[np.cos(pi*t[0]), np.cos(2*pi*t[0]),np.cos(3*pi*t[0]),np.cos(4*pi*t[0])],
#              [np.cos(pi*t[1]), np.cos(2*pi*t[1]),np.cos(3*pi*t[1]),np.cos(4*pi*t[1])],
#              [np.cos(pi*t[2]), np.cos(2*pi*t[2]),np.cos(3*pi*t[2]),np.cos(4*pi*t[2])],
#              [np.cos(pi*t[3]), np.cos(2*pi*t[3]),np.cos(3*pi*t[3]),np.cos(4*pi*t[3])]])
#print(A)
#print()
#print(A_gen)
y = np.array([3,1,-3,1])
coeff = np.linalg.solve(A_gen,y)
print('the solution of a,b,c,d: ',coeff)# 4x1 vector
#print('shape of the solution:',coeff.shape)

x = np.linspace(0,1)
#print('shape of x:',x.shape)# 50x1 vector
y_p = predict(coeff,x)
#print('shape of y:',y_p)

plt.plot(x,y_p,label='wave')
plt.plot(t,y,'rx',label='data')
plt.legend(loc='best')
plt.show()

# Second part: root finding
print()
print("Entering second part...\nDisplaying the function plot...")
# the first function
def func(x):
    return (2*(np.sin(x)**2) - 3*np.sin(x)+1)
x = np.linspace(0,pi,num=100)
y = func(x)
x_test = np.array([0,pi/6,1.5,2,3*pi/4,pi])
y_test = func(x_test)
plt.plot(x,y,label='curve')
plt.plot(x_test,y_test,'rx',label='test data')
plt.legend(loc='best')
plt.show()

root1 = fsolve(func,pi/12)
print("the root between 0 and pi/6 is",root1,"function value=",func(root1))

root2 = fsolve(func,1.75)
print("the root between 1.5 and 2 is ",root2,"function value =",func(root2))

root3 = brentq(func,3*pi/4,pi)
print("the root between 3pi/4 and pi is:",root3,"function value =",func(root3))
print()

# the second function
print("Entering second function...\nDisplaying the function plot...")
def func2(x):
    return 3*np.cos(x+1.4)

x = np.linspace(5,30,num=100)
y = func2(x)

x_test = np.array([10,15,20,25])
y_test = func2(x_test)
plt.plot(x,y,label="curve")
plt.plot(x_test,y_test,'rx',label="test data")
plt.legend(loc='best')
plt.show()

root1 = fsolve(func2,12.5)
print("the root between 10 and 15 is:",root1,"function value =",func2(root1))
root2 = fsolve(func2,22.5)
print("the root between 20 and 25 is:",root2,"function value =",func2(root2))

#using bisect and brentq
print("Now using bisect...")

root1 = bisect(func2,10,15)
print("the root between 10 and 15 is:",root1,"function value =",func2(root1))

root2 = bisect(func2,20,25)
print("the root between 20 and 25 is:",root2,"function value =",func2(root2))

print("Now using brentq...")

root1 = brentq(func2,10,15)
print("the root between 10 and 15 is:",root1,"function value =",func2(root1))

root2 = brentq(func2,20,25)
print("the root between 20 and 25 is:",root2,"function value =",func2(root2))