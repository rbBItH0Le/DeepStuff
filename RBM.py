# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:38:09 2021

@author: rohan
"""

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable

movies=pd.read_csv('movies.dat',sep='::',header=None,engine='python',encoding='latin1')
user=pd.read_csv('users.dat',sep='::',header=None,engine='python',encoding='latin1')
ratings=pd.read_csv('ratings.dat',sep='::',header=None,engine='python',encoding='latin1')

training_set=pd.read_csv('u1.base',delimiter='\t')
training_set=np.array(training_set,dtype='int') 
test_set=pd.read_csv('u1.test',delimiter='\t')
test_set=np.array(test_set,dtype='int') 

nb_users=max(max(training_set[:,0]),max(test_set[:,0]))
nb_movies=max(max(training_set[:,1]),max(test_set[:,1]))

def covert_data(drata):
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies=drata[:,1][drata[:,0]==id_users]
        id_ratings=drata[:,2][drata[:,0]==id_users]
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

training_set=covert_data(training_set)
test_set=covert_data(test_set)

training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1
training_set[training_set==0]=-1

test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1
test_set[test_set==0]=-1

class RBM():
    def __init__(self,nv,nh):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx+self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy+self.b.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)
    def train(self,v0,vk,ph0,phk):
        self.W += torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)
        
nv=len(training_set[0])
nh=100
batch_size=3 
rbm=RBM(nv,nh)

nb_epochs=10
for epoch in range(1,nb_epochs+1):
    train_loss=0
    s=0.
    for id_user in range(0,nb_users,batch_size):
          vk=training_set[id_user:id_user+batch_size]
          v0=training_set[id_user:id_user+batch_size]
          ph0,_=rbm.sample_h(v0)
          for k in range(10):
            _,hk=rbm.sample_h(vk) 
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
          phk,_=rbm.sample_h(vk)
          rbm.train(v0, vk, ph0, phk)
          train_loss+=torch.mean(torch.abs(v0[v0>0]-vk[v0>0]))
          s+=1
    print('epoch :'+str(epoch)+'loss :'+str(train_loss/s))
    
test_loss=0
s=0.
for id_user in range(nb_users):
    v=training_set[id_user:id_user+1]
    vt=test_set[id_user:id_user+1]
          ph0,_=rbm.sample_h(v0)
          for k in range(10):
            _,hk=rbm.sample_h(vk) 
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
          phk,_=rbm.sample_h(vk)
          rbm.train(v0, vk, ph0, phk)
          train_loss+=torch.mean(torch.abs(v0[v0>0]-vk[v0>0]))
          s+=1
    print('epoch :'+str(epoch)+'loss :'+str(train_loss/s))
         
        