# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

import numpy as np
import scipy
import sympy
from IPython.display import display
import matplotlib.pyplot as plt

# numerical derivatives
def forward_diff(x,y):
    all_but_last = np.diff(y) / np.diff(x)
    #print(all_but_last[-1])
    last = (y[-1] - y[-2]) / (x[-1] - x[-2])
    #print(last)
    # the last one of all_but_last is the same as last
    # append to to output y with the same length as x
    return (np.append(all_but_last,last))

def backward_diff(x,y):
    all_but_first = np.diff(y) / np.diff(x)
    first = (y[1] - y[0]) / (x[1] - x[0])
    return np.insert(all_but_first,0,first)

def central_diff(x,y):
    centre = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    first = (y[1] - y[0]) / (x[1] - x[0])
    last = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return (np.insert(np.append(centre,last),0,first))

#t = np.linspace(0.78,0.79,100)
#f = np.sin(t)
t = np.linspace(0,2*np.pi, 100)
f = np.sin(t)


dfdt_analytical = np.cos(t)
dfdt_fwd = forward_diff(t,f)
dfdt_bck = backward_diff(t,f)
dfdt_ctr = central_diff(t,f)

plt.plot(t,dfdt_fwd, label = 'forward difference approxmation')
plt.plot(t,dfdt_bck, label = 'backward difference approxmation')
plt.plot(t,dfdt_ctr, label = 'central difference approxmation')
plt.plot(t,dfdt_analytical,'k--', label = 'analytical derivative')
plt.legend()
plt.show()

# using numpy function
p = np.poly1d([1,1,1,1])
pd = np.polyder(p)
print(pd)
# we can simply using pd as a function to calculate the 
# derivative at certain location
print(pd(3.5))

# second derivative of p
print(np.polyder(p,2))

# dealing with piece-wise functions
def f1(x):
    if x<0:
        return 0
    elif 0<=x<=1:
        return x
    elif 1<=x<=2:
        return 2.0-x
    else:
        return x+1.0
print(f1(-1.0))

f2 = np.vectorize(f1)
print(f2([0,1,2,3,4]))


# integration using quad
from scipy.integrate import quad
def integrand(x):
    return x**2
ans,err = quad(integrand,0,1)
print(ans)
print(f'The estimated error is {err}')
print()
print()

# nested integrals
from scipy.integrate import dblquad
from scipy.integrate import tplquad
from scipy.integrate import nquad
# dblquad: outer --> inner
# tplquad: outer --> inner
# nquad: inner --> outer

a,b,c,d = 1, 10, 0, 0.5
f = lambda x,y:1

#Arguments:
# function,outer integral limits --> inner integral limits 
area = dblquad(f,c,d,       # y limits
               lambda x: a,# x limits
               lambda x: b)
print(area)

# function, inner integral limits --> outer integral limits
are = nquad(f,[[a,b],
               [c,d]])
print(area)
print()
print()

from sympy import solve, symbols,Symbol,Function,Eq,pprint,dsolve,Eq,sin,cos
# symbolic polynomials
a,b,c,x = symbols('a,b,c,x')
f = a*x**2 + b*x + c
print(a)
print(f)
solution = solve(f,x)
print(solution)
pprint(solution)

A,B,C = -1,2,3
x0 = solution[0].subs([(a,A),
             (b,B),
             (c,C)])
x1 = solution[1].subs([(a,A),(b,B),(c,C)])
display(x0,x1)
print()
print()

# symbolic differentiation
from sympy import diff
f = a*x**2 + b*x + c
print(diff(f,x))
print(diff(f,x,2)) # second order
print(diff(f,a)) # partial derivative
diff(f,a).subs([(x,2)])

# symbolic integration
from sympy import integrate
print(integrate(f,x))
print()
pprint(integrate(f,x))
print()
print(integrate(f,(x,0,1)).subs([(a,2),(b,2),(c,2)]))

print()

# differentiating a symbolic function
f = Function('f')
x = Symbol('x')
fd = f(x).diff(x)
pprint(fd)

# solving ODE
print('Solving ODE...')
ode = Eq(f(x).diff(x),cos(x))
gen_sol = dsolve(ode,f(x))
pprint(gen_sol)
display('Generated solution:',gen_sol)
print()
print('Solving constraints...')
# use an initial valude to find the constraint C1
cnd = gen_sol.subs([(x,0),(f(0),0)])
display(cnd)
pprint(cnd)
#cnd = Eq(gen_sol.rhs.subs(x,0),0)
print()
ode_sol = gen_sol.subs([(cnd.rhs, cnd.lhs)])
display(ode_sol)
pprint(ode_sol)
print()
# solving second order differential equation
A = Eq(f(x).diff(x,x), 12*x**2)
gen_sol = dsolve(A,f(x))
pprint(gen_sol)
display(gen_sol)
print('solving constraints...')
cnd0 = Eq(gen_sol.rhs.subs(x,0),0)
pprint(cnd0)
cnd1 = Eq(gen_sol.rhs.subs(x,1),3)
pprint(cnd1)
cnd2 = Eq(gen_sol.rhs.diff(x).subs(x,1),6)
pprint(cnd2)

C1, C2 = symbols('C1,C2')
C1C2_sol = solve([cnd0, cnd1],(C1,C2))
pprint(C1C2_sol)

# numerical solutions to ODE
from scipy.integrate import odeint
def dfdt(x,t):
    return np.cos(t)
ts = np.linspace(0,5,100)
f0 = 0
fs = odeint(dfdt,
            f0,
            ts)
fs=np.array(fs).flatten()
plt.xlabel('t')
plt.ylabel('f')
plt.plot(ts,np.sin(ts),'c',label='analytical solution')
plt.plot(ts,fs,'r--', label='numerical solution')
plt.legend(loc='best')

