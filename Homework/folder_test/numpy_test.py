# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:01:17 2018

@author: rpf19
"""

import numpy as np
#import math
#import cmath
y = np.full((1,1),3)
print(y)

z = np.arange(5,10)
print(z)
print()

z = np.arange(5,10,2)
print(z)

#linspace
z = np.linspace(-4,4,6)
print(z)

a = np.array([[0],[1],[2]])
b = np.array([0,1,2])
print("a = ",a)
print()
print("b = ",b)

new2d = np.append(a,a,axis = 0)
print(new2d)
print(f"new array shape: {new2d.shape}")

a = np.array([-2, 1, 3])
b = np.array([6, 2, 2])
c = a+b
print(c)
#
#a =np.loadtxt
c = range(1,7)
print(c)
c = 1,2,3,4,5
print(c)