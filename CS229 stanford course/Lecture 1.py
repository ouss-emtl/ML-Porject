# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:01:10 2020

@author: Rudolf
"""
import numpy as np
import pandas as pd
from math import log,exp
import matplotlib.pyplot as plt

data=pd.read_csv("exam20207.csv",sep=";")
(n,p)=data.shape
y=data['y']
X=data.iloc[:,0:p-1]

def h(theta,X):
    s=0
    for i in range(len(X)):
        
        s+=theta[i]*X[i]
    
    return s

def lms(X_train,y_train,alpha):
    #print(len(y_train))
    theta=[1 for j in range(len(y_train))]
    for i in range(len(y_train)):
        X_ray=np.array(X_train.iloc[:,i:i+1])
        theta[i]+=alpha*(y_train[i]-h(theta,X_ray))
    
    return theta

# Problem  set 1

# defining X and y



epsilon=1e-5
def g(z):
    return 1/(1+exp(-z))

def gradient(theta,X,y):
    s=0
    m=len(y)
    for i in range(m):
        s+=y[i]*(1-g(y[i]*np.dot(theta.T,X[i])))*X[i]
    return -s/m

def hessian(theta,X,y):
    s=0
    m=len(y)
    for i in range(m):
        s+=g(y[i]*np.dot(theta.T,X[i]))*(1-g(y[i]*np.dot(theta.T,X[i])))*np.dot(X[i],X[i].T)
    return s/m

def NewtonforJ(X,y):
    m=len(y)
    theta=X.shape()[1]
    while np.linalg.norm(gradient(theta,X,y)) >epsilon:
        theta+=np.dot(np.linalg.inv(hessian(theta,X,y)),gradient(theta,X,y))
    return theta

def boundaryPlot(X,y):
    X1=[];X2=[]
    for i in  range(len(y)):
        if y[i]==1:
            X1.append(X[i])
        else:
            X2.append(X[i])
    
    plt.scatter(X1[:1,],X1[1:2,], marker='^')
    plt.scatter(X2[:1,],X2[1:2,], marker='o')
    # boundary line
    theta=NewtonforJ(X,y)
    def theboundary(x1):
        return -(theta[0]-0.5+theta[1]*x1)/theta[2]
    x1=np.linspace(0,20)
    y1=[theboundary[x] for x in x1]
    plt.plot(x1,y1)
    plt.show()
    