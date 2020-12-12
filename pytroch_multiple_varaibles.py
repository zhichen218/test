#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 22:51:06 2020

@author: zhichen
"""

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim


### Load the data

# First we load the entire CSV file into an m x 3
D = torch.tensor(pd.read_csv("/home/zhichen/Documents/pytorch/linreg-multi-synthetic-2.csv", header=None).values, dtype=torch.float)

# We extract all rows and the first 2 columns, and then transpose it
x_dataset = D[:, 0:2].t()

# We extract all rows and the last column, and transpose it
y_dataset = D[:, 2].t()

# And make a convenient variable to remember the number of input columns
n = 2


### Model definition ###


means = x_dataset.mean(1, keepdim=True)
deviations = x_dataset.std(1, keepdim=True)






# First we define the trainable parameters A and b 
A = torch.randn((1, n), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Then we define the prediction model
#def model(x_input):
#    return A.mm(x_input) + b


def model(x_input):
    x_transformed = (x_input - means) / deviations
    return A.mm(x_transformed) + b



### Loss function definition ###

def loss(y_predicted, y_target):
    return ((y_predicted - y_target)**2).sum()

### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
optimizer = optim.Adam([A, b], lr=0.001)

# Main optimization loop
for t in range(200000):
    # Set the gradients to 0.
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    y_predicted = model(x_dataset)
    # See how far off the prediction is
    current_loss = loss(y_predicted, y_dataset)
    # Compute the gradient of the loss with respect to A and b.
    current_loss.backward()
    # Update A and b accordingly.
    optimizer.step()
    print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")