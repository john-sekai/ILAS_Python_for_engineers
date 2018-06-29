# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:36:48 2018

@author: rpf19
"""
import numpy as np
import matplotlib.pyplot as plt
# Simple Plot
x = np.linspace(-3,3,100)
y = x**3 - x**2 - 4*x + 4.0
print(len(y))
plt.plot(x,y,'r--')
for i in range(0,len(y)):
    if y[i] == 0.0:
        plt.plot(x[i],y[i],'ks',markersize=10)
# Because linspace cannot always gets the root in its array 
# using polynomial to get the roots and plot them
p = np.poly1d([1,-1,-4,4])
print(p.r)
plt.plot(p.r,p(p.r),'ks',markersize=10)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Plot of Polynomial: $x^3-x^2-4x+4$')
plt.savefig("sample_data\\simpleplot-pengfei.png")
plt.show()

