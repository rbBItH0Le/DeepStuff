# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:56:30 2021

@author: rohan
"""

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies=pd.read_csv('movies.dat',sep='::',header=None,encoding='latin-1',engine='python')
users=pd.read_csv('users.dat',sep='::',header=None,encoding='latin-1',engine='python')


training_set=pd.read_csv('u1.base',delimiter='\t')
training_set=np.array(training_set,dtype=int)
test_set=pd.read_csv('u1.test',delimiter='\t')
test_set=np.array(test_set,dtype=int)

max_user=max(max(training_set[:,0]),max(test_set[:,0]))
max_movies=max(max(training_set[:,1]),max(test_set[:,1]))


def convert_data(data):
    dum=[]
    for i in range(1,max_user+1):
        id_movies=data[:,1][data[:,0]==i]
        id_ratings=data[:,2][data[:,0]==i]
        rartings=np.zeros(max_movies)
        rartings[id_movies-1]=id_ratings
        dum.append(list(rartings))
    return dum
        
training_set=convert_data(training_set)
test_set=convert_data(test_set)

training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)


class SAE(nn.Module):
    def __init__(self,):
        super(SAE,self).__init__()
        self.fc1=nn.Linear(max_movies,20)
        self.fc2=nn.Linear(20,10)
        self.fc3=nn.Linear(10,20)
        self.fc4=nn.Linear(20,max_movies)
        self.activation=nn.Sigmoid()
    def forward(self,x):
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.fc4(x)
        return x
    
sae=SAE()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(sae.parameters(),lr=0.01,weight_decay=0.5)

nb_epochs=200
for i in range(1,nb_epochs+1):
    training_loss=0
    s=0.
    for j in range(max_user):
        input=Variable(training_set[j]).unsqueeze(0)
        target=input.clone()
        if torch.sum(target.data>0)>0:
            output=sae(input)
            target.require_grad=False
            output[target==0]=0
            loss=criterion(output,target)
            mean_corrector=max_movies/float(torch.sum(target.data>0)+1e-10)
            loss.backward()
            training_loss+=np.sqrt(loss.data*mean_corrector)
            s+=1.
            optimizer.step()
    print("the epoch number is"+str(i)+"and the train loss is"+str(training_loss/s))
    
test_loss=0
s=0.
for j in range(max_user):
   input=Variable(training_set[j]).unsqueeze(0)
   target=Variable(test_set[j]).unsqueeze(0)
   if torch.sum(target.data>0)>0:
      output=sae(input)
      target.require_grad=False
      output[target==0]=0
      loss=criterion(output,target)
      mean_corrector=max_movies/float(torch.sum(target.data>0)+1e-10)
      test_loss+=np.sqrt(loss.data*mean_corrector)
      s+=1.
print("and the test loss is"+str(test_loss/s))
            
            
