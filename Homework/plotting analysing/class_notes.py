# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:09:51 2018

@author: rpf19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d

plot_flag = False
if plot_flag == True:
    # scatter plot 
    x = [-1, 3, 4, 8, 10]
    f = [-1, -2, 7, 13, 1]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax1.plot(x, f, '-rx', linewidth = 3,label = "data 1")
    
    plt.legend(loc='best', fontsize = 12)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$f$', fontsize=20)
    plt.title("Simple plot of $f$ against $x$", fontsize =18)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(121)
    num_points = 100
    x = np.linspace(0, 4*np.pi, num_points)
    f = np.sin(x)
    
    # Plot graph
    ax2.plot(x,f)
    plt.xlabel('$x$')
    plt.ylabel('$\sin(x)$')
    
    plt.savefig("my-plot.png")
    
    # try myself
    x = np.linspace(0,4*np.pi, num_points)
    y1 = np.square(np.sin(x))
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    x_marker = np.zeros(8)
    for i in range(0,8):
        x_marker[i] = 0.5 * (i+1) * np.pi
    print(x_marker)
    y1_marker = np.square(np.sin(x_marker))
    
    ax3.plot(x,y1,x_marker,y1_marker,'rx', markersize=4)

    #plot data
    A = np.loadtxt('C:\\Users\\rpf19\\ILAS-python-backup\\ILAS_Python_for_engineers\\sample_data\\sample_data.dat')
    print(A)
    print((type(A)))
    print(A[0][1])

    B = np.loadtxt('C:\\Users\\rpf19\\ILAS-python-backup\\ILAS_Python_for_engineers\\sample_data\\sample_student_data.txt', delimiter="\t",dtype=str)
    print(type(B))
    print(B)
    
    C = np.loadtxt('C:\\Users\\rpf19\\ILAS-python-backup\\ILAS_Python_for_engineers\\sample_data\\sample_student_data.txt', delimiter="\t",dtype=float,skiprows=3, usecols=(3,4))
    print(type(C))
    print(C)
    students = C

    from scipy.stats import linregress
    
    h = students[:,0]
    w = students[:,1]
    
    # start linear regression
    m,c = linregress(h,w)[0], linregress(h,w)[1]
    print('m = ',m,'\n')
    print('c = ',c,'\n')
    wfit = m * h + c
    plt.plot(h,w,'bo', label='experiment data')
    plt.plot(h,wfit, label='fit')
    
    plt.xlabel('x')
    plt.ylabel('y')

    # polynomial fit
    x = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    y = np.array([50.0, 38.0, 30.6, 25.6, 22.2, 19.5, 17.4])*1e-3
    
    d = np.polyfit(x,y,2)
    yfit = np.poly1d(d)(x)
    plt.plot(x,yfit,'r-')
    plt.plot(x,y,'b-')
    
    RMSE = lambda raw,fitted: np.sqrt(np.sum((fitted - raw)**2) / len(y))
    rmse2 = RMSE (y,yfit)

    print('RMSE = ', rmse2)

    r = 0.0123456789
    s = 0.12345432
    print(s)
    
    # cast as string
    print('%s' % r)
    
    print('{}'.format(r))
    
    # cast as string using scientific notation
    print('%.3E' % r)
    print('%.2E, %.1E' % (r,s))

    # fitting an arbitrary function
    x = [0.000000000000000000e+00, 1.052631578947368363e+00, 2.105263157894736725e+00, 3.157894736842105310e+00,4.210526315789473450e+00, 5.263157894736841591e+00, 6.315789473684210620e+00,7.368421052631578760e+00,8.421052631578946901e+00,9.473684210526315042e+00,1.052631578947368318e+01,1.157894736842105132e+01,1.263157894736842124e+01,1.368421052631578938e+01,1.473684210526315752e+01, 1.578947368421052566e+01,1.684210526315789380e+01,1.789473684210526372e+01,1.894736842105263008e+01,2.000000000000000000e+01]
    y = [7.445192947240600745e+01, 4.834835792411828947e+01, 6.873305436340778840e+01, 5.979576407972768948e+01,6.404530772390434379e+01,6.090548420541189500e+01, 7.157546008677115879e+01, 8.620253336570679892e+01, 1.138154622045899913e+02, 8.493639813028174501e+01, 9.783457330550828601e+01, 1.082064229481453594e+02, 1.063876210674365979e+02, 1.001971993955305038e+02, 1.061496321788094832e+02, 1.279575585921491836e+02, 1.556956405962417875e+02, 1.584164804859289859e+02, 1.753888794716459358e+02, 1.980941276403034124e+02]
    #print(type(x))
    
    RMSE = lambda raw,fitted: np.sqrt(np.sum((fitted - raw)**2) / len(y))
    # cast list to ndarray
    x = np.array(x)
    y = np.array(y)
    
    plt.plot(x,y,'bx', label='raw data')
    
    def exponential(x,a,b):
        y = a* np.exp(b*x)
        return y
    ####========CORE CODE================
    from scipy.optimize import curve_fit
    
    opt, cov = curve_fit(exponential, x, y)
    ####========CORE CODE================
    y_fit = exponential(x,*opt)
    rmse = RMSE(y_fit,y)
    
    plt.plot(x,y_fit,'r', label='fit')
    plt.legend(loc='best')
    plt.title('RMSE:{rmse}')
    
    print(f"y = {opt[0]} * e**({opt[1]}*x)")
    
    from scipy.interpolate import interp1d
    x = np.array([0.5,1,3,4])
    y = np.array([1.648,  2.718, 20.086, 54.598])
    
    func = interp1d(x,y)
    y_fit = func(x)
    plt.plot(x,y,'bx')
    plt.plot(x,y_fit,'-r')
    
    from scipy.interpolate import splrep,splev
    import scipy.interpolate
    
    spl = splrep(x,y, k=2)
    x2 = np.linspace(0,10,100)
    y2 = splev(x2,spl)
    plt.plot(x,y,'bo',x2,y2,'g-')
    plt.show()
    
    x_int = np.arange(0.5, 4.1, 0.1)
    y_init = np.exp(x_int)
    
    # function data
    plt.plot(x_int,np.exp(x_int), 'k--',label ='y=exp(x)')
    order_poly = 2
    
    # interpolated data - SWITCH ORDER OF x AND y
    func = splrep(y, x, k=order_poly)
    x_int = splev(y_init,func)
    plt.plot(x_int, y_init,'g--', label=f'polynomial order {order_poly}')
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    # get coefficients that define ppolynomial curve
    F = scipy.interpolate.PPoly.from_spline(func)
    
    # break points or knots
    print('F.x = ',F.x,'\n')
    # coefficients at breakpoints
    print(F.c)

    # HISTOGRAM
    # loc = mean, scale = stddev, size = num_samples
    x = np.random.normal(loc = 0.0, scale = 1.0, size = 4000)
    n,bins,patches =plt.hist(x,20, facecolor='green')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.show()
    print('bins = ',bins,' \npatches = ',patches,'\n')
    
    # VISUALISING 2D ARRAYS
    x = np.array([[8,7,6,8],
                  [8,7,6,3],
                  [6,6,5,2],
                  [4,3,2,1]])
    plt.matshow(x, cmap = cm.Wistia)
    plt.colorbar()
    plt.show()


# 3D PLOTTING
from mpl_toolkits.mplot3d import axes3d
N = 50
x = np.linspace(-np.pi, np.pi, N)
y = np.linspace(-np.pi, 2*np.pi, N)
X,Y = np.meshgrid(x,y)
#print(np.around(X,2),np.around(Y,2))

f = np.sin(X)*np.cos((X*Y**2)/10)

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X,Y,f,cmap=cm.Reds)
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('Z', fontsize=20)







