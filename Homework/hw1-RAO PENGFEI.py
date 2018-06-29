# -*- coding: utf-8 -*-
"""
Spyder Editor

Test yourself
PART A
Based on the flow chart(chart is different from
the table) 
"""
MarketRate = 0.0091
JPY = 50_000_000
print("PART A:")
print("Amount in JPY sold:",JPY)

if JPY < 10_000:
    multiplier = 0.9
elif JPY < 100_000:
    multiplier = 0.925
if JPY >= 100_000 and JPY < 1_000_000:
    multiplier = 0.95
if JPY >= 1_000_000 and JPY < 10_000_000:
    multiplier = 0.97
if JPY > 10_000_000:
    multiplier = 0.98
    
USD = JPY * MarketRate * multiplier
EffectiveRate = USD / JPY

print("Amount in USD purchased:" , USD)
print("Effective rate:" , EffectiveRate)   
        
"""
PART B
"""
CashFlag = True;
CashMultiplier = 1.0
if CashFlag == True:
    CashMultiplier = 0.9
USD = USD * CashMultiplier
EffectiveRate = USD / JPY

print("\nPART B:")
print("Amount in JPY sold:",JPY)
print("Amount in USD purchased:" , USD)
print("Effective rate:" , EffectiveRate) 

"""
Classifier
"""
for i in range(1,30,2):
    if i % 3 == 0:
        print(i," belongs to group A")
    elif i % 4 == 0:
        continue
    elif i < 20 and i > 5:
        print(i," belongs to group B")
    else:
        print(i," belongs to group C")
    