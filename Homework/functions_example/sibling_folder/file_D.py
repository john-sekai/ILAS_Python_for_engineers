# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:02:39 2018

@author: rpf19
"""

#import sys
#sys.path.append('../')
#sys.path.append("C:/Users/rpf19/AppData/Local/Programs/Python/Python36/Lib/site-packages/tensorflow")

#sys.path.remove('C:/Users\rpf19\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\tensorflow')
#print(sys.path)
import tensorflow as tf
hello = tf.constant("Hello World!")
def Hello(hello):
    print(hello)
sess = tf.Session()
sess.run(Hello(hello))