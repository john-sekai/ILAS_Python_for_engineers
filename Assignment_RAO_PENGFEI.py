# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:42:25 2018

@author: rpf19
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
#　======================= Q1 =================================
print(f'Now importing data from the following url: {url}')
data = pd.read_csv(url,delimiter='|')
print('Displaying data...')
print(data.head())
print()
print('Counting each occupation...')
print(data['occupation'].value_counts())

# ======================= Q2 ==================================
print('\nNow entering Q2...')
data_clean = data[data.occupation!='other']
data_clean = data_clean[data_clean.occupation!='retired']
#data_clean.reindex(index=np.arange(len(data_clean)))
print(data_clean.head())
print('Counting each occupation...')
print(data_clean['occupation'].value_counts())
print()

# ======================= Q3 ==================================
print('\nNow entering Q3:\nNow visualizing the occupation counts...')
job_data = data_clean['occupation'].value_counts() # job_data is a series

# extract the occupations and their counts
job_list = job_data.index.values.tolist()[0:10] # select the top 10 occupations
# add 'other' for bar plot
job_list.append('other') 
job_count = job_data.values.tolist() # occupation counts 
# calculate the number of other occupation
other = np.sum(job_count[10:]) # the count of other occupations
job_count = job_count[0:10]
job_count.append(other)# append the count of other occupations

# print the top 10 occupations
print(f'The top 10 occupations are:\n{job_list[0:10]}')
print(f'the count of other occupation is: {other}')

print('Now plotting bar plot...')
# create new plot
plt.figure(figsize=(15,10))
# create the position for displaying occupations in bar plot
x_pos = np.arange(len(job_count))
# bar plot 
plt.bar(x_pos,job_count)
# display the occupation names
plt.xticks(x_pos,job_list,rotation=45,fontsize=20)

# adjusting some parameters
plt.yticks(fontsize=20)
plt.xlabel('Occupation',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.title('Statistics of occupations',fontsize=35)
plt.show()

# ==================== Q4 ===============================
print('\nNow entering Q4...')
admin = data_clean[data_clean.occupation == 'administrator']
print(admin.head())
print(f'The mean age of administrator:{admin.age.mean()}')
print(f'The standard deviation age of administrator:{admin.age.std()}')

# ================ Entering Part B =======================
print('Now entering Part B...')
# ========================== Q1 ========================== 
print('Now entering Q1...')
t = np.linspace(0,2*np.pi,100)
T1 = 1.0
T2 = 1.0/3.0
t_mark = [0,np.pi/2,np.pi*2]
rx = lambda t: np.exp(-np.divide(t,T1))
ry = lambda t: 2*np.cos(np.divide(t,T2))
plt.figure(figsize=(15,10))
plt.plot(rx(t),ry(t),'k-')
plt.plot(rx(t_mark),ry(t_mark),'ro',markersize=10)
for ti in t_mark:
    plt.annotate(f't = {ti}',xy = (rx(ti),ry(ti)), xytext = (rx(ti)+0.02,ry(ti)+0.05),fontsize=15)
plt.show()

# =========================== Q2 ==========================
print('Now entering Q2...')
print('Q2 part a:')
from sympy import Symbol,diff,pprint
from scipy.misc import derivative
import sympy as sp
t = Symbol('t')
r_x = sp.exp(-t/T1)
r_y = 2*sp.cos(t/T2)
a_x = (r_x).diff(t,t)
a_y = (r_y).diff(t,t)
pprint(f'the second order derivative of rx: {a_x}')
pprint(f'the second order derivative of ry: {a_y}')
print(a_x.subs(t,0.1))

print('Q2 part b:')
dt = 100
tspan = np.linspace(0,2*np.pi,dt).tolist()
at = np.zeros(dt)

# use the result from the sympy diff() to calculate the true acceleration
def ax_ref(t):
    return np.exp(-1.0*np.asarray(t))
def ay_ref(t):
    return -18.0*np.cos(3.0*np.asarray(t))
at_ref = np.sqrt(np.add(np.square(ax_ref(tspan)),np.square(ay_ref(tspan))))

# use the numpy and scipy numerical method to calculate the derivative
def rx(t):
    return np.exp(np.divide(-t,T1))
def ry(t):
    return np.multiply(2.0,np.cos(np.divide(t,T2)))
a = lambda t:np.sqrt(np.add(np.square(derivative(rx,t,dx=1e-5,n=2)), np.square(derivative(ry,t,dx=1e-5,n=2))))

at[0] = a(0.0)
for t_index in np.arange(1,dt):
    at[t_index] = a(tspan[t_index])

#plot the absolute acceleration
plt.figure(figsize = (15,10))
plt.plot(tspan,at,'k-')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Acceleration Magnitude',fontsize=20)
plt.title('Numerical Differentiation Method',fontsize=20)
plt.show()

plt.figure(figsize = (15,10))
plt.plot(tspan,at_ref,'r-')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Acceleration Magnitude',fontsize=20)
plt.title('Direct Method',fontsize=20)
plt.show()
#============================ Q3 ==========================
from sympy.solvers.solveset import nonlinsolve
print('Now entering Q3...') 
print('Part A:')
v_x = np.array((sp.exp(-t/T1)).diff(t).subs(t,1.0)).astype(np.float64)
v_y = np.array((2*sp.cos(t/T2)).diff(t).subs(t,1.0)).astype(np.float64)
v = np.sqrt(np.add(np.square(v_x),np.square(v_y)))
print(f'The magnitude velocity of the particle at time = 1s is:{v} m/s')

print('Part B:')
t_sol = nonlinsolve([r_y],[t])
print('The time t when ry = 0 are:')
pprint(t_sol)
#============================ Q4 ==========================
print('Now entering Q4...')
import matplotlib.animation as animation
print('Animating the trajectory of the particle...')
# 100 frames
t = np.linspace(0,100)
x = np.exp(-t/T1)
y = 2*np.cos(t/T2)
fig = plt.figure(figsize = (15,10))
ax = plt.axes(xlim=(-0.5,1.5),ylim=(-3,3))
s = 0
def animate(i):
    # Timestep = 1/20 = 0.05
    i /= 20
    if i == 0:
        s = i
    else:
        s = np.linspace(0,i)
    x = np.exp(-s/T1)
    y = 2*np.cos(s/T2)
    ax.plot(x,y,'r')
# 50ms delay between frames to match timestep
ani = animation.FuncAnimation(fig, animate ,frames = 100, interval = 50)
ani

# save animation
writer = animation.writers['ffmpeg'](fps=15,bitrate=1800)
ani.save('assignment.mp4',writer = writer)
