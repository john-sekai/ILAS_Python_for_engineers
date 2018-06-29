# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:39:42 2018

@author: rpf19
"""
import sys
import numpy as np
if(not "..\\" in sys.path):
    sys.path.append("..\\")
import my_library as mylib

print(mylib.my_func(3,4))

# lambda functions
func_list = [lambda x : np.cos(x), lambda x : np.sin(x), lambda x : np.tan(x)]

for i in range(1,4):
    print("for i = " + str(i) + " :\n")
    for func in func_list:
        print(func(i))
        print()