'''
Created on Jan 24, 2019

@author: AnneMarie
'''
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.duration.tests.test_survfunc import test_weights1

def closed_form(X,y):
    X_tr = np.transpose(X)
    weights_cf = np.linalg.inv(X_tr@X)@X_tr@y
    return weights_cf


def gradient_descent(X,y,wo,b,n,epsilon):
    X_tr = np.transpose(X)
    X_mult = X_tr@X
    XY_mult = X_tr@y
    diff = np.ones((2,1))
    i=1
    while diff.max()>epsilon:
        a = n/(1+b*i)
        weights = wo - 2*a*(X_mult@wo-XY_mult)
        i=i+1
        diff = abs(weights-wo)   
        wo = weights
        #if i>100000:
        #    break
    return weights





#X1 = [[0.86,1],[0.09,1],[-0.85,1],[0.87,1]
   #   ,[-0.44,1],[-0.43,1],[-1.10,1]
    #  ,[0.40,1],[-0.96,1],[0.17,1]]

#y1 = [[2.49],[0.83],[-0.25],[3.10],[0.87],
     #[0.02],[-0.12],[1.81],[-0.83],[0.43]]
#wo = [[1],[1]]
#b = 100
#n =20
#epsilon = 10**-6

#weights1 = closed_form(X1,y1)
#y_hat1 = np.transpose(X1@weights1)
#y_hat2 = weights1[0]*[row[0] for row in X1] + weights1[1]





