__author__ = 'jr'

import numpy as np
import xgboost as xgb
import tsne


def loaddata(file):
    data = np.loadtxt(file, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?'), 94: lambda x:int(x[6])-1 } )
    data = data[1:,]
    return data

trainfile = '/home/jr/Documents/Kaggle/Otto Group/TrainingData/train.csv'
testfile = '/home/jr/Documents/Kaggle/Otto Group/TestData/test.csv'
tnsefile = '/home/jr/Documents/Kaggle/Otto Group/TrainingData/tsne.csv'

train = np.loadtxt(trainfile, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?'), 94: lambda x:int(x[6])-1 } )[:,1:]
test = np.loadtxt(testfile, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?') } )[:,1:]

y = train[:,-1]
x = np.concatenate((train[:,:-1],test), axis = 0)

x = 1/(1+np.exp(-np.sqrt(x)))

#mytsne = tsne.bh_sne(x)
#np.savetxt(tnsefile, mytsne, delimiter=',', newline='\n')
mytnse = np.loadtxt(tnsefile, delimiter=',')

np.round(x, 3)

param = {}
param['objective'] = 'multi:softprob'
param['eval_metric'] = 'mlogloss'
param['num_class'] = 9
param['max_depth'] = 8
param['eta'] = .03
param['min_child_weight'] = 3
num_round = 2

x = np.append(x, mytnse, axis=1)

trainlen = y.shape[0]

trainX = x[:(trainlen),:]
testX = x[(trainlen):,:]

xgtrain = xgb.DMatrix(trainX, label=y)
bst = xgb.train(param, xgtrain, num_round)
