# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:00:30 2018

@author: rpf19
"""
import sys
sys.path.remove('..s')
sys.path.append('..\\')
print(sys.path,end='\n\n')
from functions_example import sibling_folder as sf
import functions_example.sibling_folder.file_C
print(sf.file_C.subtract_and_increment(8,10))