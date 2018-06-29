# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:12:43 2018

@author: RAO PENGFEI
"""
import numpy as np

# dot product
C = [-1, 2, 6]
D = [4, 3, 3]
prod = np.dot(C, D)
print("The dot product =",prod)

# calculate the angle between C and D
C_mag = np.linalg.norm(C)
D_mag = np.linalg.norm(D)
print("Magnitude of C =",C_mag)
print("Magnitude of D =",D_mag)
theta = np.arccos(prod/(C_mag*D_mag))
theta = np.degrees(theta)
print("the angle between C and D =",theta,"(degree)")
if prod > 0:
    print("the angle between A and B is acute")
elif prod < 0:
    print("the angle between A and B is obtuse")
else:
    print("the angle between A and B is right angle")

# importing data
np.set_printoptions(suppress=True)
A = np.loadtxt('ILAS_Python_for_engineers\sample_data\douglas_data.csv',
               delimiter=",",usecols=(1,2,3,4,5,6,7,8),dtype=float,skiprows=2)
#print("imported data A=",A)
B = A[0:10][0::1]
#print(B)

# manipulating data
for i in range(0,10):
    B[i][7] *= 1e6
#print("Now B =",B)

# calculate mass of each beam
sec_a = 100*1e-4
print(sec_a)
mass = np.zeros((10,1))
for j in range(0,10):
    mass[j] = sec_a * (B[j][5] * 1e-2) * B[j][4]    
B = np.append(B,mass,axis=1)
print(B)

# display mass
print("The mass of beam 1 is ",B[0][-1])
print(f"The mass of beam 1 is {B[0][-1]}")
print(f"{B[4][0::2]}")