# -*- coding: utf-8 -*-
"""
Documentation String
Created on Mon May 14 16:37:33 2018

@author: rpf19
"""

my_list = []
def my_function(r):
    print(r)
    r += 2
    print(r)
    my_list.append(r)
    
for r in range(5):
    my_function(r)

print(my_list)
def subtract_and_increment(a,b):
    "Return a minus b ,plus 1"
    c = a- b +1
    return c
print(subtract_and_increment(a = 3, b= 5), subtract_and_increment(a =5, b = 3))