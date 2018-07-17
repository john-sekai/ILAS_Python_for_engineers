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
print("with sort method:",lab_group0)

# insert method
lab_group0.insert(0,"Mark")
print("using insert method",lab_group0)

# pop method
lab_group0.pop(-1)
print("pop the last item",lab_group0)

# nested list
lab_group0 = ["Sara","Mika","Ryo","Am"]
lab_group1 = ["Hemma","Miri","Quy","Sajid"]
lab_group2 = ["Adam","Yukari","Farad","Fumitoshi","Rao"]
lab_group = [lab_group0, lab_group1, lab_group2]
print(lab_group)

# iterating over lists
for d in lab_group:
    print("the value is:",d)
for d in lab_group:
    for i in d:
        print("Group member name is:",i)
        
# iterate to cast each item to string type
data = [1, 2.0, "three"]
print("data = ",data)
for j in data:
    j=str(j)
    print(j)

# completely cast data to the exact string
data = str(data)
print("casted data = ",data)
print("type of data",data,type(data))
for k in data:
    print(k)

# enumerate()
lab_group0 = ["Sara", "Mari", "Quang", "Sam", "Ryo", "Nao", "Takashi"]
a = enumerate(lab_group0)
b = list(enumerate(lab_group0))
print("a = ",a)
print("b = ",b)
for l in b:
    print("in b, l =",l)
    print("type of l:",type(l))
a_h = hash(a)
print("hash of a = ",a_h)

