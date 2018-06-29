# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:45:57 2018

@author: rpf19
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# A NUMPY way to load data
A = np.loadtxt('sample_data/sample_student_data.txt', dtype = float, skiprows=9,
               usecols=(3,4))
print(A)
print()

# basic operation of pandas
# ADD A FRAME to data
print("==========dataframe in pandas===========")
data = np.random.randint(0,10, size=(4,4))
print(data)
print()
print(pd.DataFrame(data))
print()

indices = ["Student 1", "Student 2", "Student 3","Student 4"]
headers = ["score1","score2","socre3","score4"]

data_frame = pd.DataFrame(data, index = indices, columns = headers)
print(data_frame)
print()

# input data using pandas
students = pd.read_csv('sample_data/sample_student_data.csv',skiprows=[1],index_col = 0)
print(students)
print()
print()
from IPython.display import display
display(students[:4]) # display the first 4 entries
display(students.head(4)) # display first 4 entries
display(students.tail(4))
print()
print()

# pd.read_csv can also read from urls
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, sep='\t')
display(chipo.head())
print()
print()

#
Location = 'sample_data/noheader_noindex.xlsx'
df = pd.read_excel(Location)
display(df.head())

# for unlabelled data
# first method is to omit heads
unlabelled = pd.read_csv('sample_data/noHeader_noIndex.csv', header=None)
display(unlabelled)

# the second method is add a header yourself
headers = ["X","Y"]
unlabelled_vert = pd.read_csv('sample_data/noHeader_noIndex_vert.csv',names=headers)
display(unlabelled_vert)
print()
print()

# rename cols names
capitalised = students.rename(columns={'Height':'HEIGHT',
                                       'Weight':'WEIGHT'}, inplace=False)
print(students.head())
print(capitalised.head())
print()
print()

# rename index
nums = students.rename(index={'JW-1':'1',
                       'JW-10':'10'},inplace=False)
display(nums.head())
print()
print()

students.index.names = ['STUDENTS']
display(students.head())
print()
print()

students.describe()
print()
print()

print(students.mean())
print()
#calculate the correlation between columns
print(students.corr())
print()
# count data sample
print(students.count())
print()

print(students.max())
print()
print(students.min())
print()
print(students.median())
print()
print(students.std())
print()

# Selecting data
print("============Selecting Data=============")
students = pd.read_csv('sample_data/sample_student_data.csv',skiprows = [1], index_col = 0)

# columns can be selected using their keywords, just like dictionaries
print(students['Weight'].head())
print()
print(students['Weight'][:5])
print()
print(students[['Weight','DOB']][:5])
print()
# this is an alternative method for single column
print(students.Weight.head())

print(students['Weight'].head())
print(students[:5].Weight)
print()

print("==============Indexing=============")
print(students)
print()
print(students.iloc[0,1])
print()
# usful when using certain column as indexing column
print("------using loc-------")
print(students.loc['JW-1','Sex'])
print()
print(students.loc['JW-1'])
print()
print(students.loc[:,'Height'])
print()
print(students.Weight.mean())
print()
print(students.loc[:,['sex','BP']].count())
print()
print(students.loc[['JW-3','JW-5','JW-10'], 'Height'].mean())
print('\n\n\n\n')

print(students.iloc[2:6,2:4].max())
print(students.iloc[:,2:4].min())
print()
print("=========Time Series Data===========")
students = pd.read_csv('sample_data/sample_student_data.csv',skiprows=[1],index_col=0,parse_dates=[2])
print()
print()
print()

print("======== numpy functions for data frames=========")
#print('Maximum height amongst students',    students.Height.max())
#print('Tallest student:',                   students.Height.argmax())
#print('Sex of tallest student:',            students.Sex[students.Height.argmax()])

print("\n\n\n\n\nCalculating BMI...\n")
BMI = np.divide(students['Height']**2, students['Weight'])
print(BMI)

print("========= Boolean data frame indexing ============")
girls = students.index[students.Sex =='F']

taller_than = students.index[students.Height > 1.6]
girls_taller_than = students.index[((students.Sex == 'F') * (students.Height > 1.6))]
print('All female students:',girls.tolist(), end='\n\n')
print('All students taller than 1.6m:', taller_than.tolist(), end = '\n\n')
print('All female students taller than 1.6m:',girls_taller_than.tolist(), end='\n\n')
print(girls, end = '\n\n\n')


print("========= Applying a function to the whole series=========")
def square(x):
    return x**2

students.Weight = students.Weight.apply(square)
print(students.head(), end = '\n\n\n\n')

print("========= Sorting data frames ===========")
print('Students in height order:\n')
print(students.sort_values(by='Height',ascending=False).head())
print()

print("========= Data Cleaning =================")
header = ['Item 1', 'Item 2', 'Item 3','Item 4']

df = pd.read_csv('sample_data/data_with_holes.csv', names = header)
print(df)
print()
# finding NaN in data
print(df.isnull())
print()
print(df.isnull().sum())
print()
print(df.notnull())
print()
print()

# methods to remove missing values
print("remove rows with NaN:\n")
# remove rows
print(pd.read_csv('sample_data/data_with_holes.csv', names = header).dropna())
print()

# remove columns
print("remove columns with NaN:\n")
print(pd.read_csv('sample_data/data_with_holes.csv', names = header).dropna(axis=1))
print()

# fill NaN with 0
print("fill NaN with 0:\n")
print(pd.read_csv('sample_data/data_with_holes.csv', names = header).fillna(0))
print()

# fill gaps with mean of others values in column
print("fill NaN with mean in the column:\n")
df = pd.read_csv('sample_data/data_with_holes.csv', names = header)
print(df.fillna(df.mean()))
print()

#df = pd.read_csv('sample_data/data_with_holes.csv', names = header)
students = pd.read_csv('sample_data/sample_student_data.csv', 
                    skiprows=[1],   
                    index_col=0,
                    parse_dates=[2])

for index,row in students.iterrows():
    print(index, row["DOB"])
print()
plt.plot(students.Weight, students.Height,'o')
plt.show()

print(students.keys())
print('\n\n\n')

students.plot('Height','Weight', kind="scatter")
plt.show()

print("========= Adding a column to a data frame ===========")
students["BloodGroup"] = ['B','A','O','B','O',
                          'AB','O', 'B', 'A',
                          'B','A', 'A', 'O', 'O', 'A', 
                          'O', 'O', 'A', 'O', 'B', 'B', 'B']
print(students.head())
del students["BloodGroup"]
print("-------- Add a column using calculation from other columns--------")
students.insert(loc=1,column='BMI', value=(students.Height**2 / students.Weight))
print(students)
del students['BMI']
students.loc['JW-8'] = ['M', '17/04/1996',1.69, 55.0, '121/82']
students.loc['JW-13'] = ['M', '17/04/1996',1.69, 55.0, '121/82']
print(students.tail())

print()
print("================= drop rows and columns =================")
students = pd.read_csv('sample_data/sample_student_data.csv',skiprows=[1],index_col=0)
display(students.head())
students_drop = students.drop(index=['JW-1','JW-2','JW-4'], columns=['DOB'], inplace=False)
print("inplace means whether the modification will happen to the self object:\n",students_drop.head())
print()
print("inplace false helps protect the original data:\n",students.head())
students.drop(index=['JW-1','JW-2','JW-4'], columns=['DOB'], inplace=True)
display(students.head())
print()
print()

# reset index will overwrite the original indexing columns
students.reset_index(drop=True,inplace=True)
print(students.head())

rain = pd.read_csv('sample_Data/rotterdam_rainfall_2012.txt',
                   skiprows=9,
                   parse_dates=['YYYYMMDD'],
                   index_col='YYYYMMDD',
                   skipinitialspace=True)
display(rain.head())

rain.RH[rain.RH < 0] = 0 #remove negative values
display(rain.head())
print()

monthlyrain = rain.RH.resample('M',
                             kind='period'
                             ).sum()
print(monthlyrain)
monthlyrain.plot(kind='bar')
plt.ylabel('mm/month')
plt.xlabel('month')
plt.show()

monthlyrain.to_csv('sample_data/rotterdam_monthly_rain_formatted.csv',
                   header=True,
                   float_format="%.3f",
                   sep="\t",
                   mode="a",
                   index=True)
print()
print("============== file read and write================")
file = open("sample_data/my_file.txt","w")
print("File Name:", file.name)
print("Open Mode:", file.mode)

with open("sample_data/twister.txt","w") as file:
    file.write("How can you clam cram in a clean cream can?")
with open("sample_data/twister.txt","r+") as file:
    print(f"Position in File Now:{file.tell()}")
    
    twister = file.read()
    
    print(twister)
    position = file.seek(31)
    file.write("caravan?")
    
    print(f"Position In File Now:{file.tell()}") 
    file.truncate()
    position = file.seek(0)
    twister = file.read()
    print(twister)
    
print("\n\n\n")
with open("sample_data/twister.txt" , "r+" ) as file : 
    
    # tell 
    print(f"Position In File Now:{file.tell()}") 
    

    # reading the file moves the position
    twister =  file.read()
    print(twister)
    print(f"Position In File Now:{file.tell()}") 
    
    
    # seek
    position =  file.seek( 31 )
    print(f"Position In File Now:{file.tell()}") 
    
    
    # write starting at new position
    file.write("caravan?")
    
    # deletes all characters from current position onwards
    file.truncate()    
    
    # move back to start poistion
    position =  file.seek( 0 )
    print(f"Position In File Now:{file.tell()}") 
    
    # print updated string
    twister =  file.read()
    print(twister)