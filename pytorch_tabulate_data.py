#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:00:59 2020

@author: zhichen
"""


# dataset and a way to iterate over it to produce batches of features/labels
# model
# training/evaluation loop



#https://averdones.github.io/reading-tabular-data-with-pytorch-and-training-a-multilayer-perceptron/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


#  ---------------  Dataset  ---------------

class StudentsPerformanceDataset(Dataset):      # this is in heritage
#class StudentsPerformanceDataset(csv_file):    
    
    
    """Students Performance dataset."""

    def __init__(self, csv_file):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        df = pd.read_csv(csv_file)

        # Grouping variable names
        self.categorical = ["gender", "race/ethnicity", "parental level of education", "lunch",
                           "test preparation course"]
        self.target = "math score"

     #   print (self.target )
        
     #   print (self.categorical)
        # One-hot encoding of categorical variables
        #self.students_frame = pd.get_dummies(df, prefix=self.categorical)
        
        self.students_frame = pd.get_dummies(df, prefix=self.categorical)

        print (self.students_frame)
        
        type(self.students_frame)
        
        # Save target and predictors
        self.X = self.students_frame.drop(self.target, axis=1)
        self.y = self.students_frame[self.target]

        print (self.X)
        print (self.y)
        
    def __len__(self):
        return len(self.students_frame)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]



class RegressionColumnarDataset(Dataset):
    def __init__(self, df, cats, y):
        self.dfcats = df[cats]
        self.dfconts = df.drop(cats, axis=1)
        
        self.cats = np.stack([c.values for n, c in self.dfcats.items()], axis=1).astype(np.int64)
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32)
        self.y = y.values.astype(np.float32)
        
    def __len__(self): return len(self.y)
 
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]


#  ---------------  Model  ---------------

class Net(nn.Module):

    def __init__(self, D_in, H=15, D_out=1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)     #H is 
        self.fc2 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x.squeeze()


#  ---------------  Training  ---------------

def train(csv_file, n_epochs=100):
    """Trains the model.
    Args:
        csv_file (str): Absolute path of the dataset used for training.
        n_epochs (int): Number of epochs to train.
    """
    # Load dataset
    dataset = StudentsPerformanceDataset(csv_file)

    #print("total sample #",len(dataset))
    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])


    #print("total sample #",len(dataset),"train_size",train_size,"test_size",test_size)
    # Dataloaders
    trainloader = DataLoader(trainset, batch_size=200, shuffle=True)
    testloader = DataLoader(testset, batch_size=200, shuffle=False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
   # D_in, H = 19, 15       #where the 19 coming from ?????
    
    D_in, H = 19, 15       #where the 19 coming from ?????
    net = Net(D_in, H).to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)

    # Train the net
    loss_per_iter = []
    loss_per_batch = []
    for epoch in range(n_epochs):

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            print (inputs.float())
            outputs = net(inputs.float())
            
            print (outputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        loss_per_batch.append(running_loss / (i + 1))
        running_loss = 0.0

    # Comparing training to test
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs.float())
    print("Root mean squared error")
    print("Training:", np.sqrt(loss_per_batch[-1]))
    print("Test", np.sqrt(criterion(labels.float(), outputs).detach().cpu().numpy()))

    # Plot training loss curve
    plt.plot(np.arange(len(loss_per_iter)), loss_per_iter, "-", alpha=0.5, label="Loss per epoch")
    plt.plot(np.arange(len(loss_per_iter), step=4) + 3, loss_per_batch, ".-", label="Loss per mini-batch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import os
    import sys
    import argparse

    # By default, read csv file in the same directory as this script
    csv_file = os.path.join("/home/zhichen/Documents/pytorch/StudentsPerformance.csv")

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", nargs="?", const=csv_file, default=csv_file,
                        help="Dataset file used for training")
    parser.add_argument("--epochs", "-e", type=int, nargs="?", default=100, help="Number of epochs to train")
    args = parser.parse_args()

    # Call the main function of the script
    train(args.file, args.epochs)