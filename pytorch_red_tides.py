#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:00:59 2020

@author: zhichen
"""


#https://averdones.github.io/reading-tabular-data-with-pytorch-and-training-a-multilayer-perceptron/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
   
import os
import sys

from sklearn.preprocessing import StandardScaler 



#  ---------------  Dataset  ---------------

class RedTidesDataset(Dataset):      # this is in here 
#class StudentsPerformanceDataset(csv_file):    
    
    
    """Students Performance dataset."""  #triple quote 

    def __init__(self, csv_file):   # input variable here 
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        #df = pd.read_csv(csv_file)
        
#        xy=np.loadtxt(csv_file,delimiter=',',dtype=np.float32,skiprows=1)
#
#        self.x=torch.from_numpy(xy[:,0:5])
#        self.y=torch.from_numpy(xy[:,[7]])
#        self.n_sample=xy.shape[0]
#        
        #print(self.x)
        #print(self.y)
    

        df = pd.read_csv(csv_file)
        x=df.iloc[:,0:5].values
        y=df.iloc[:,[7]].values
        
       
        sc=StandardScaler()

        x_train=sc.fit_transform(x)
        y_train=sc.fit_transform(y)
        
        self.x=torch.tensor(x_train,dtype=torch.float32)    #convert into tensor and redefine the dtype
        self.y=torch.tensor(y_train,dtype=torch.float32)
        self.n_sample=df.shape[0]

     
    
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index],self.y[index]

        

#  ---------------  Model  ---------------

class Net(nn.Module):

    def __init__(self, D_in, H=15, D_out=1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)       # D_in mean umber of features, or independent varaibles
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, D_out)      # d_out dependent variable, or out ot classes 
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

       # return F.log_softmax(x,dim=1)
        return x

#  ---------------  Training  ---------------

def train(csv_file, n_epochs=100):
    """Trains the model.
    Args:
        csv_file (str): Absolute path of the dataset used for training.
        n_epochs (int): Number of epochs to train.
    """
    # Load dataset
    dataset = RedTidesDataset(csv_file)

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size=50, shuffle=True)   #len(traninset)/batch_size=len(trainloader)
    testloader = DataLoader(testset, batch_size=50, shuffle=False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    D_in, H = 5, 5
    net = Net(D_in, H).to(device)


   # inputSize = len(trainDataset.columns) -1
    # Loss function
    #criterion = nn.NLLLoss()
    
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Train the net
    loss_per_iter = []
    loss_per_batch = []
    
    for epoch in range(n_epochs):

        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):   #i interation 
            inputs = inputs.to(device)
            labels = labels.to(device)


          #  print (inputs)
          #  print (labels)
            # Zero the parameter gradients
            optimizer.zero_grad()

            #net.zero_grad()
            
            # Forward + backward + optimize
            #print("eppch",epoch, "interation",i, "batch size", len(inputs))
            #print (inputs.float())
            outputs = net(inputs)
            
            #outputs = net(inputs.view(-1,50*5))
            
           # print(outputs)
           # print (labels)
            loss =criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # Save loss to plot
           # print("epoch",epoch, "interation",i, "loss",loss)
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        print ("epoch",epoch, "loss",loss)
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
#    plt.plot(np.arange(len(loss_per_iter), step=4) + 3, loss_per_batch, ".-", label="Loss per mini-batch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
 
    
    import argparse

    # By default, read csv file in the same directory as this script
    csv_file = os.path.join("/home/zhichen/Documents/pytorch/sorted_same_day_data_v2.csv")

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", nargs="?", const=csv_file, default=csv_file,
                        help="Dataset file used for training")
    parser.add_argument("--epochs", "-e", type=int, nargs="?", default=100, help="Number of epochs to train")
    args = parser.parse_args()

    # Call the main function of the script
    train(args.file, args.epochs)
    
    
    
#preds = []
#with torch.no_grad():
#   for val in X_test:
#       y_hat = model.forward(val)
#       preds.append(y_hat.argmax().item())   
    
# df = pd.DataFrame({'Y': y_test, 'YHat': preds})
#df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]   
    
#df['Correct'].sum() / len(df)    