# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:42:25 2018

@author: rpf19
"""

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
#　Q1
print(f'Now importing data from the following url: {url}')
data = pd.read_csv(url,delimiter='|')
print('Displaying data...')
print(data.head())
print()
print('Counting each occupation...')
print(data['occupation'].value_counts())
print('\nNow entering Q2...')
# Q2
data_clean = data[data['occupation']!='other']
data_clean = data_clean[data_clean['occupation']!='retired']
print(data_clean.head())
print(data_clean['occupation'].value_counts())
# Q3
