# -*- coding: utf-8 -*-
"""
Created on Mon May 14 17:35:37 2018

@author: rpf19
"""

# hydrostatic pressure
def hydrostatic_pressure(h,*,  rho = 1000.0, g = 9.81):
    """
    compute the hydrostatic pressure using rho, g and h
    rho: kg m-3
    g: m s-2
    h: m
    """
    P = rho * g * h
    return P