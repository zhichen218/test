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
import pickle



from sklearn.preprocessing import StandardScaler 





def noaa_oc3 (x):

    
    Rrs443=x[:,0] 
    Rrs486=x[:,1] 
    Rrs551=x[:,2] 


    R1=Rrs443/Rrs551
    R2=Rrs486/Rrs551

    chla=np.empty(shape=(len(R1)))
    chla[:]=0.0
    
    for i in range (len(R1)):
        ratio=np.log10(max([R1[i],R2[i]]))
        tmp=0.2228-2.4683*ratio+1.5867*ratio*ratio-0.4275*ratio*ratio*ratio-0.7768*ratio*ratio*ratio*ratio
        chla[i]=np.power(10,tmp)
    
    return chla



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
      #  x=df.iloc[:,0:5].values
      #  y=df.iloc[:,[7]].values
        
        x=df.iloc[:,[9,12,16]].values
 #       y=df.iloc[:,[9]].values   #chla 
        
        y=df.iloc[:,[6]].values   #chla  counts
        
        
#        depths=df.iloc[:,[5]].values
#        
#        idx=np.where(depths<10)
#        
#        idx=idx[0]
#        
#        x=x[idx,:]
#        y=y[idx]
#        
        x=np.log10(x)
        y=np.log10(y)
        
      
        #y=df.iloc[:,[8]].values    # cell_count
        
        sc=StandardScaler()

        x_train=sc.fit_transform(x)
        y_train=sc.fit_transform(y)
        
        
        y_mean=np.mean(y)
        y_std=np.std(y)
        
       # y_std_scale=(y-y_mean)/y_std
        
        
        x_means=np.mean(x,axis=0)
        x_stds=np.std(x,axis=0)
        
        #x_std_scale=(x-x_means)/x_stds#column means
        
        
        with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([y_mean,y_std,x_means,x_stds], f)
        
        
        #x_train=x
        #y_train=y
        
        #
        x_train=x
        y_train=y
        
        self.x=torch.tensor(x_train,dtype=torch.float32)    #convert into tensor and redefine the dtype
        self.y=torch.tensor(y_train,dtype=torch.float32)
        self.n_sample=len(self.y)

     
    
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index],self.y[index]

        

#  ---------------  Model  ---------------

class Net(nn.Module):

    def __init__(self, D_in, H=3, D_out=1):
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
#
#def train(csv_file, n_epochs=1000):
#    """Trains the model.
#    Args:
#        csv_file (str): Absolute path of the dataset used for training.
#        n_epochs (int): Number of epochs to train.
#    """
#    # Load dataset
#    dataset = RedTidesDataset(csv_file)
#
#    # Split into training and test
#    train_size = int(0.80 * len(dataset))
#    test_size = len(dataset) - train_size
#    trainset, testset = random_split(dataset, [train_size, test_size])
#
#    # Dataloaders
#    trainloader = DataLoader(trainset, batch_size=20, shuffle=True)   #len(traninset)/batch_size=len(trainloader)
#   # testloader = DataLoader(testset, batch_size=20, shuffle=False)
#    
#    testloader = DataLoader(testset, batch_size=test_size,shuffle=False)
#
#    # Use gpu if available




#
#    # Define the model
#    D_in, H = 3, 3
#    net = Net(D_in, H).to(device)
#
#
#   # inputSize = len(trainDataset.columns) -1
#    # Loss function
#    #criterion = nn.NLLLoss()
#    
#    criterion = nn.MSELoss()
#    #criterion = nn.CrossEntropyLoss()
#
#    # Optimizer
#    optimizer = optim.Adam(net.parameters(), lr=0.005)
#
#    # Train the net
#    loss_per_iter = []
#    loss_per_batch = []
#    
#    for epoch in range(n_epochs):
#
#        running_loss = 0.0
#        
#        for i, (inputs, labels) in enumerate(trainloader):   #i interation 
#            inputs = inputs.to(device)
#            labels = labels.to(device)
#
#
#          #  print (inputs)
#          #  print (labels)
#            # Zero the parameter gradients
#            optimizer.zero_grad()
#
#            #net.zero_grad()
#            
#            # Forward + backward + optimize
#            #print("eppch",epoch, "interation",i, "batch size", len(inputs))
#            #print (inputs.float())
#            outputs = net(inputs)
#            
#            #outputs = net(inputs.view(-1,50*5))
#            
#           # print(outputs)
#           # print (labels)
#            loss =criterion(outputs, labels)
#            
#            loss.backward()
#            optimizer.step()
#
#            # Save loss to plot
#           # print("epoch",epoch, "interation",i, "loss",loss)
#            running_loss += loss.item()
#            loss_per_iter.append(loss.item())
#
#        print ("epoch",epoch, "loss",loss)
#        loss_per_batch.append(running_loss / (i + 1))
#        running_loss = 0.0
#
#
#
#
#    # save trained model into the 
#    torch.save(net.state_dict(),'chl_net_wfl.pt')
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
D_in, H = 3, 3
net = Net(D_in, H).to(device)    
    
net.load_state_dict(torch.load('chl_net_wfl'))
net.eval()




    
    ### how to save the model parameter out 
    # Comparing training to test
dataiter = iter(testloader)
    
    #dataiter = iter(trainloader)
inputs, labels = dataiter.next()
inputs = inputs.to(device)
labels = labels.to(device)
outputs = net(inputs.float())




inputs_xx=inputs.detach().cpu().numpy()


with open('objs.pkl', 'rb') as f:  
        y_mean,y_std,x_means,x_stds=pickle.load(f)


inputs_xx2=inputs_xx*x_stds+x_means

inputs_xx2=np.power(10,inputs_xx)

noaa_chla=noaa_oc3(inputs_xx2)



out_yy=np.ndarray.flatten(outputs.detach().cpu().numpy())
out_xx=np.ndarray.flatten(labels.detach().cpu().numpy())

  #  print (np.corrcoef(out_xx,out_yy))

#    out_xx2=out_xx*y_std+y_mean
#    out_yy2=out_yy*y_std+y_mean
#    
    
out_xx2=np.power(10,out_xx)

out_yy2=np.power(10,out_yy)


fig, ax = plt.subplots()

print (np.corrcoef(out_xx2,out_yy2))

plt.plot(out_xx2,out_yy2,marker='o',markersize=10,label="Pytorch Model",linestyle = 'None')

  

plt.plot(out_xx2,noaa_chla,marker='o',markersize=10,label="NASA OC3 Model",linestyle = 'None',
         markerfacecolor='none')

plt.yscale('log')
plt.xscale('log')

plt.xlim([0.01,100])
plt.ylim([0.01,100])

plt.xticks(fontsize=12,fontweight="bold")
plt.yticks(fontsize=12,fontweight="bold")

#plt.set_xticklabels(fontweight="bold")

#plt.set_yticklabels(fontweight="bold")

  #  plt.ticklabel_format (fontweight="bold")

print (np.corrcoef(out_xx2,noaa_chla))
h2=plt.legend(fontsize=12)

plt.legend(loc=2)


plt.ylabel("Predicted Chl a (mg/m$^3$)",fontweight="bold")
plt.xlabel("Measured Chl a (mg/m$^3$)",fontweight="bold")


plt.show()


plt.savefig('scatter_plt.png', bbox_inches='tight')

#save validation out 

dd=np.column_stack((out_xx2,out_yy2,noaa_chla))

dd2=pd.DataFrame(dd)

dd2.to_csv ("validation_results.csv")




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
#csv_file = os.path.join("/home/zhichen/Documents/pytorch/sorted_same_day_data_v2.csv")

csv_file = os.path.join("/home/zhichen/Documents/pytorch/sorted_WFS_nomad.csv")

df = pd.read_csv(csv_file)

x=df.iloc[:,[9,12,16]].values
 #       y=df.iloc[:,[9]].values   #chla 
    
y=df.iloc[:,[6]].values   


noaa_chla=noaa_oc3(x)

#plt.plot(y,noaa_chla,'o')

# Parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument("--file", "-f", nargs="?", const=csv_file, default=csv_file,
                    help="Dataset file used for training")
parser.add_argument("--epochs", "-e", type=int, nargs="?", default=1500, help="Number of epochs to train")
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