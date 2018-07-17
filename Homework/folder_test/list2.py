# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
lab_group0 = ["Sara","Mari","Quang"]
size = len(lab_group0)
print("Lab group members:", lab_group0)
print("Size of lab group;",size)
print("Check the Python object type:", type(lab_group0))
# sorted function
names = sorted(lab_group0)
print("the sorted lab_group0",names)
# sort method
lab_group0.sort()
print(lab_group0)
# insert method
lab_group0.insert(0,"Mark")
print(lab_group0)
# pop method
lab_group0.pop(-1)
print(lab_group0)