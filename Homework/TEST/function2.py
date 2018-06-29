# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:35:43 2018
to test the usage of function in Python
@author: rpf19
"""

def test_args_func(first, second ,third, fourth):
    "Prints each argument of the function"
    print(first, second, third, fourth)
dictionary = {"second":12, "fourth":4, "third":3}

# '*' means argument
# '**' means keyward argument
# using ** means to take in as keyward input
# the following code generate different outputs
test_args_func(1,*dictionary) # output the names of the dictionary 
test_args_func(1,**dictionary) # output the values of the dictionary
