# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:15:11 2021

@author: rohan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Credit_Card_Applications.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler(feature_range=(0,1))
X=mn.fit_transform(X)

from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X,num_iteration=100)

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar() 
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
show()

fra=som.win_map(X)
frauds=fra[(1,6)]
frauds=mn.inverse_transform(frauds)
