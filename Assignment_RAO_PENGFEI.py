# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:42:25 2018

@author: rpf19
"""
import numpy as np
import matplotlib.pyplot as plt
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
print('Counting each occupation...')
print(data_clean['occupation'].value_counts())
print()

# ======================= Q3 ==================================
print('Now visualizing the occupation counts...')
job_data = data_clean['occupation'].value_counts() # job_data is a series

# extract the occupations and their counts
job_occu = job_data.index.values # occupation list
job_count = job_data.values # occupation counts

# print the top 10 occupations
print('The top 10 occupations are:')
print(job_occu[0:10])

# calculate the number of other occupation
print('the count of other occupation is:')
other = np.sum(job_count[10:]) # the count of other occupations
print(other)

x_pos = np.arange(len(job_data.values[0:10])+1)
plt.figure(figsize=(15,10))
print(x_pos)
print(np.concatenate(job_occu[0:10], ('other'))
#job_count_bar = job_count[0:10]
#job_count_bar.append(other)
plt.bar(x_pos,job_count[0:10])


plt.bar(10,other)
plt.xticks(x_pos,job_occu[0:10],rotation=45,fontsize=20)
#plt.xticks([10],'other',rotation=45,fontsize=20)

plt.yticks(fontsize=20)
plt.xlabel('Occupation',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.title('Statistics of occupations',fontsize=35)
plt.show()

# ==================== END of Q3 ===============================