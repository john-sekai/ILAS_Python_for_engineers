# -*- coding: utf-8 -*-
"""
Created on Sun May  6 22:58:24 2018

@author: rpf19
"""
# import everything in nltk.book
from nltk.book import *

runflag = 0
if runflag == 1:
    # display concordance, insensitive to case
    text1.concordance("whales")
    text2.concordance("affection")
    text3.concordance("lived")
    text4.concordance("nation")
    text5.concordance("lol")
    
    print()
    print()
    # find words used in similar contexts
    text1.similar("monstrous")
    print()
    text2.similar("monstrous")
    
    # find contexts that are shared by several words
    print()
    text2.common_contexts(["monstrous","very"])
    
    # display the dispersion plot of word(s)
    text4.dispersion_plot(["citizens","China","humanity","responsibility"])
    
    # generate random text in styles, seems not available
    #text2.generate("monstrous")
    #print()
    #text3.generate()

# calculate the distribution frequency of the words
fdist1 = FreqDist(text1) 
print(fdist1)
fdist1.most_common(50)
fdist1['whales']