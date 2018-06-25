# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:22:10 2018

@author: rpf19
"""
import numpy as np
# root finding
from sympy import solve,Eq,symbols,pprint
a,b,x = symbols('a,b,x')
f = a*x**3 - b*x**2 - a*x + b
print('Solving function:')
eq = Eq(f,0)
pprint(eq)
sol = solve(f,x)
print('...')
print('The solutions are:')
pprint(sol[0])
pprint(sol[1])
pprint(sol[2].subs([(a,3),(b,1)]))

# Part2: car braking
print('Entering second part...')
from scipy.integrate import quad
v0 = 15
tao = 2
def velocity(t):
    return v0*np.exp(-t/tao)
dist = quad(velocity,0,10)
print(f'Travelled distance: {dist[0]} m')