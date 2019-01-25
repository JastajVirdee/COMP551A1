'''
Created on Jan 24, 2019

@author: AnneMarie
'''
import numpy as np


def closed_form(X,y):
    X_tr = np.transpose(X)
    weights_cf = np.linalg.inv(X_tr@X)@X_tr@y
    return weights_cf

X1 = [[0.86,1],[0.09,1],[-0.85,1],[0.87,1]
      ,[-0.44,1],[-0.43,1],[-1.10,1]
      ,[0.40,1],[-0.96,1],[0.17,1]]

y1 = [[2.49],[0.83],[-0.25],[3.10],[0.87],
     [0.02],[-0.12],[1.81],[-0.83],[0.43]]

weights1 = closed_form(X1,y1)

print(weights1)