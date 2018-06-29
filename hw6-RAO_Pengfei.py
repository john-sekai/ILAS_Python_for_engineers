# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:36:48 2018

@author: rpf19
"""
import numpy as np
import matplotlib.pyplot as plt
# ================Simple Plot=====================
# to create the range of x as polynomial input
x = np.linspace(-3,3,100)

# calculate the output y to plot
y = x**3 - x**2 - 4.0*x + 4.0

# plot with red dash line
plt.plot(x,y,'r--')

# figure out if the x includes some of the roots
for i in range(0,len(y)):
    if y[i] == 0.0:
        plt.plot(x[i],y[i],'ks',markersize=10)

# Unfortunately, there is only one root in x and y.
# Because linspace cannot always gets the root in the 
# linspace array, so I use 1d-polynomial object to 
#get the roots and plot them
p = np.poly1d([1,-1,-4,4])
#print(p.r)
plt.plot(p.r,p(p.r),'ks',markersize=10)

# add the refinement
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Plot of Polynomial: $x^3-x^2-4x+4$')
plt.savefig("sample_data\\pengfei-simple_plot.png")

# fix the plot as a final one so that following plots
# are plotted in a new plot object 
plt.show()
# =============END OF Simple Plot=================

# =============Curve Fitting======================
data = np.loadtxt("sample_data\signal_data.csv",delimiter=',')
#print(data.shape)
x = data[0][:]
y = data[1][:]

# define the function to be optimized
def sinoid_func(x,a,b):
    return a*np.sin(x+b)

# define the root mean square error function
rmse = lambda raw, fitted:np.sqrt(np.sum((raw-fitted)**2) / len(raw))

# perform the optimization
from scipy.optimize import curve_fit
opt, cov = curve_fit(sinoid_func, x, y)
y_fit = sinoid_func(x,*opt)

# display the result
plt.plot(x,y,'b',label='raw data')
plt.plot(x,y_fit,'r',label = 'fit data')
plt.legend(loc = 'best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('RMSE = '+str(rmse(y,y_fit)))
print('fitted equation:\ny = '+str(opt[0])+' * sin(x+'+str(opt[1])+')')
plt.savefig("sample_data\\pengfei-curve_fitting.pdf")
plt.show()
# =============END of Curve Fitting===============

# =============Interpolation======================
# input the data
d = [0,     -1,  -2,   -3,    -4,  -5,    -6,  -7,    -8,   -9, -10]
t = [19.1, 19.1, 19.0, 18.8, 18.7, 18.3, 18.2, 17.6, 11.7, 9.9, 9.1]

# plot the raw data
plt.plot(d,t,'bx',label='raw_data')
from scipy.interpolate import splrep,splev
import scipy.interpolate

# reverse the data before interpolation, because the 
# function only takes x input from small numbers to larger numbers
d.reverse()
t.reverse()

# interpolation using first order interpolation, finding out the coefficients 
spl2 = splrep(x=d,y=t, k=1)

# data to generate interpolation plot
d2 = np.linspace(-10,0,100)
t2 = splev(d2,spl2)

# interpolation using first order interpolation, finding out the coefficients 
spl3 = splrep(d,t,k=2)

# data to generate interpolation plot
d3 = d2
t3 = splev(d3,spl3)
plt.plot(d2,t2,'r',d3,t3,'y',label=['first-order','second-order'])

# predict the temperature at 7.5m depth
y2 = splev(-7.5,spl2)
y3 = splev(-7.5,spl3)
plt.plot(-7.5,y2,'go')
plt.plot(-7.5,y3,'gx')
print('the temperature at 7.5m depth is '+ str(y2)+' degree C with linear interpolation' )
print('the temperature at 7.5m depth is '+ str(y3)+' degree C with quadratic interpolation' )
plt.show()

# the quadratic interpolation is much better in continuity, without sudden turn
# in derivative, the second order interpolation basically works better
print('Basically, the result from quadratic interpolation is more creditble')





