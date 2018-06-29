# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:58:31 2018

@author: rpf19
"""

# part one hydrostatic pressure
def hydrostatic_pressure(h,g=9.81,*,rho =1000.0):
    """
    doc-string
    calculate the hydrostatic pressure
    input:
    submerged depth h (m)
    accerleration g: default to 9.81 (m/s2)
    density of the liquid rho, default to 1000 (kg/m3), must be inputted as 
        a keyword argument
    output:
        pressure (pa)
    """
    return rho*g*h

p = 0.0
# submerged at a depth of 10m
p = hydrostatic_pressure(10.0)
print("submerged at a depth of 10m, pressure = ",p)

# submerged at a depth of 10m at the equator
p=hydrostatic_pressure(10.0, 9.78)
print("submerged at a depth of 10m at the equator, pressure = ",p)

# submerged in sea water at a depth of 10m at the equator
p=hydrostatic_pressure(10.0, 9.78, rho = 1022.0)  
print("submerged in sea water at a depth of 10m at the equator, pressure = ",p)

# part 2 currency trading
def currency_trading(JPY,CashFlag):
    """
    calculate the USD 
    input:
        JPY: amount of Japanese Yen to be traded
        CashFlag: to indicate that the trading is finished in cash
    output:
        USD: the amount of USD purchased
        EffectiveRate: the actual amount of USD per JPY
    """
    MarketRate = 0.0091
    
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
    
    CashMultiplier = 1.0
    if CashFlag == True:
        CashMultiplier = 0.9
    USD = USD * CashMultiplier
    EffectiveRate = USD / JPY
    
    return USD, EffectiveRate
 
# call the function
JPY = 50_000_000
CashFlag = True

USD, EffectiveRate = currency_trading(JPY, CashFlag)

print("Amount in JPY sold:",JPY)
print("Amount in USD purchased:" , USD)
print("Effective rate:" , EffectiveRate) 