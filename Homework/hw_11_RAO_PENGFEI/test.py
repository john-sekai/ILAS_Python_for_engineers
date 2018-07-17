# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:40:03 2018

@author: rpf19
"""

import pytest
import program_RAO

from program_RAO import *

assert absolute(1) == 1
assert absolute(-1) == 1

with pytest.raises(ValueError):
    absolute('10')