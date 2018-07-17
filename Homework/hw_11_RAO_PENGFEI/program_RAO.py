# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:00:18 2018

@author: rpf19
"""
import pytest
import numbers

def absolute(num):
    # return the absolute value of the number
    if isinstance(num,numbers.Number):
        return abs(num)
    else:
        raise ValueError('Input must be a number.')