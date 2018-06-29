# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 22:12:47 2018

@author: rpf19
"""

import pandas as pd
from IPython.display import display

data = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv",
                   sep='\t')
print("Imported data:",data)
#print(type(data))

print()

# print the names of all columns
print("The names of all columns:") 
display(data.columns)

print()
# remove duplicate items
(data.drop_duplicates(subset = 'item_name',keep = 'first',inplace = True))
data.reset_index(drop=True, inplace=True)
print("data after dropping duplicate items:")
display(data)

#Remove all items without a choice_description from the data set
print("data after dropping items without a choice description:")
print(data.dropna())

print()
# cast all prices as float data
print("Casting prices to float...")
data['item_price'] = data['item_price'].str.lstrip('$').astype('float')
print(data['item_price'])

# creating new data with 'item_name' and 'item_priceipo'
header = ['item_name' 'item_priceipo']
print("generating data with item_name and item_priceipo...")
new_data = pd.DataFrame(data.item_name)
new_data['item_priceipo'] = data['item_price']/data['quantity']
display(new_data)

print()
# sorting the data in decending order
print("Sorting the priceipo in descending order...")
sorted_data = new_data.sort_values(by = "item_priceipo", ascending = False)
sorted_data.reset_index(drop=True, inplace=True)
print(sorted_data)

#save the data in a csv file
sorted_data.to_csv("Sorted_data_RAO_Pengfei.csv", 
                   header=True,
                   float_format="%.2f",
                   sep="\t",
                   mode="a",
                   index=False)