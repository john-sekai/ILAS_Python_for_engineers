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
print()
# ======================= Q2 ==================================
print('\nNow entering Q2...')
print('Removing occupation: other and retired...')
# remove the 2 required occupations
data_clean = data[data.occupation!='other']
data_clean = data_clean[data_clean.occupation!='retired']
print('the clean data:')
print(data_clean.head())
print('Counting each occupation...')
print(data_clean['occupation'].value_counts())
print()
# ======================= Q3 ==================================
print('\nNow entering Q3:\nNow visualizing the occupation counts...')
# get the count of each occupation
job_data = data_clean['occupation'].value_counts() # job_data is a series
# select the top 10 occupations as list
job_list = job_data.index.values.tolist()[0:10] 
# add occupation:'other' for bar plot
job_list.append('other') 
# occupation counts as list
job_count = job_data.values.tolist() 
# calculate the number of other occupation
other = np.sum(job_count[10:]) 
# get the count of top 10 occupations from original data
job_count = job_count[0:10]
# append the count of other occupations
job_count.append(other)

# print the top 10 occupations
print(f'The top 10 occupations are:\n{job_list[0:10]}')
# print the count of other occupation
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

# adding sticks, labels and title to the plot
plt.yticks(fontsize=20)
plt.xlabel('Occupation',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.title('Statistics of occupations',fontsize=35)
# show the plot before entering Q4
plt.show()
# ==================== Q4 ===============================
print('\nNow entering Q4...')
# get the data with occupation the same as adminstrator
admin = data_clean[data_clean.occupation == 'administrator']
print(admin.head())
# answer the question about mean and standard variance
print(f'The mean age of administrator:{admin.age.mean()}')
print(f'The standard deviation age of administrator:{admin.age.std()}')
print()
# ================ Entering Part B =======================
print('Now entering Part B...')
# ========================== Q1 ========================== 
print('Now entering Q1...')
# time series
t = np.linspace(0,2*np.pi,100)
# some parameters given by the question
T1 = 1.0
T2 = 1.0/3.0
# the time stamp for display
t_mark = [0,np.pi/2,np.pi*2]
# the lambda function of the position of the particle
rx = lambda t: np.exp(-np.divide(t,T1))
ry = lambda t: 2*np.cos(np.divide(t,T2))
# generate the plot with size (15,10)
plt.figure(figsize=(15,10))
# plot the trajectory of the particle
plt.plot(rx(t),ry(t),'k-')
# plot the marker for certain position
plt.plot(rx(t_mark),ry(t_mark),'ro',markersize=10)
# add some annotation to the plot, specifying the location of the annotation
for ti in t_mark:
    plt.annotate(f't = {ti}',xy = (rx(ti),ry(ti)), xytext = (rx(ti)+0.02,ry(ti)+0.05),fontsize=15)
# display the plot before entering Q2
plt.show()
# =========================== Q2 ==========================
print('Now entering Q2...')
print('Q2 part a:')
from sympy import Symbol,diff,pprint
from scipy.misc import derivative
import sympy as sp
# specify the variable t
t = Symbol('t')
# claim the symbol function for derivative
r_x = sp.exp(-t/T1)
r_y = 2*sp.cos(t/T2)
# calculate second derivative
a_x = (r_x).diff(t,t)
a_y = (r_y).diff(t,t)
# display the result
pprint(f'the second order derivative of rx: {a_x}')
pprint(f'the second order derivative of ry: {a_y}')
print()
print('Q2 part b:')
# specify the time step
dt = 100
# time span
tspan = np.linspace(0,2*np.pi,dt).tolist()
# initialize the result of the derivative method
at = np.zeros(dt)

# use the displayed result from part a to directly calculate the true acceleration
# specify the acceleration
def ax_ref(t):
    return np.exp(-1.0*np.asarray(t))
def ay_ref(t):
    return -18.0*np.cos(3.0*np.asarray(t))
at_ref = np.sqrt(np.add(np.square(ax_ref(tspan)),np.square(ay_ref(tspan))))

# use the numpy and scipy numerical method derivative() to calculate the derivative
# specify the position function
def rx(t):
    return np.exp(np.divide(-t,T1))
def ry(t):
    return np.multiply(2.0,np.cos(np.divide(t,T2)))
# calculate the derivative of pos function rx and ry, and return the acceleration function
a = lambda t:np.sqrt(np.add(np.square(derivative(rx,t,dx=1e-5,n=2)), np.square(derivative(ry,t,dx=1e-5,n=2))))
# calculate the acceleration across the time span
for t_index in np.arange(dt):
    at[t_index] = a(tspan[t_index])
#plot the absolute acceleration
plt.figure(figsize = (15,10))
plt.plot(tspan,at_ref,'r-')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Acceleration Magnitude',fontsize=25)
plt.title('Direct Method',fontsize=30)
print('Plotting the acceleration plot...')
# the second plot,according to the second calculation method
plt.figure(figsize = (15,10))
plt.plot(tspan,at,'k-')
plt.xlabel('Time(s)',fontsize=20)
plt.ylabel('Acceleration Magnitude',fontsize=25)
plt.title('Numerical Differentiation Method',fontsize=30)
plt.show()
#============================ Q3 ==========================
print('Now entering Q3...') 
print('Part A:')
from sympy.solvers.solveset import nonlinsolve
# evalute the velocity at t = 1 and convert it to numpy float
v_x = np.array((sp.exp(-t/T1)).diff(t).subs(t,1.0)).astype(np.float64)
v_y = np.array((2*sp.cos(t/T2)).diff(t).subs(t,1.0)).astype(np.float64)
# calculate the magnitude of the velocity
v = np.sqrt(np.add(np.square(v_x),np.square(v_y)))
print(f'The magnitude velocity of the particle at time = 1s is:{v} m/s')

print('Part B:')
# solve the nonlinear function of the r_y = 0 with the variable t
t_sol = nonlinsolve([r_y],[t])
# pretty print the solutions as sets
print('The time t when ry = 0 are:')
pprint(t_sol)
#============================ Q4 ==========================
print('Now entering Q4...')
import matplotlib.animation as animation
print('Animating the trajectory of the particle...')
# 100 frames
frame_num = 100
# 50ms per frame
interval = 50
t = np.linspace(0,frame_num)
# claim the numpy function of position
x = np.exp(-t/T1)
y = 2*np.cos(t/T2)
# create the figure
fig = plt.figure(figsize = (15,10))
# create the coordinate axes and specify the limit of the x and y axis
ax = plt.axes(xlim=(-0.5,1.5),ylim=(-3,3))
# claim the animate function
def animate(i):
    # Timestep = 1/20 = 0.05
    i /= (1000/interval)
    # update the s for creating motion
    if i == 0:
        s = i
    else:
        s = np.linspace(0,i)
    # create the trajectory using s
    x = np.exp(-s/T1)
    y = 2*np.cos(s/T2)
    # plot the trajectory in the coordinate
    ax.plot(x,y,'r')
# 50ms delay between frames to match timestep
ani = animation.FuncAnimation(fig, animate ,frames = frame_num, interval = interval)
# display the animation
ani
# save animation
writer = animation.writers['ffmpeg'](fps=15,bitrate=1800)
ani.save('assignment.mp4',writer = writer)