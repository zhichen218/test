#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:54:18 2020

@author: zhichen
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

training_inputs=np.array([[0,0,1],
                         [1,1,1],
                         [1,0,1],
                         [0,1,1]])
    
training_outputs=np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights=2*np.random.random((3,1))-1

print ("first guessed weigth")
print (synaptic_weights)
    
for interation in range (10000):
    input_layer=training_inputs

    #forward propogation
    outputs=sigmoid(np.dot(input_layer,synaptic_weights))
    
    #print ("output ")
    #print (output)
    
    #back propogation
    
    error=training_outputs-outputs
    
    print (interation, error)
    
    adjustments=error*sigmoid_derivative (outputs)
    
    synaptic_weights +=np.dot(input_layer.T,adjustments)



print ("                     ")
print ("synaptic weight after train")
print (synaptic_weights)


print ("output after train")
print (outputs)

    

