__author__ = 'jr'

import numpy as np
import xgboost as xgb
import tsne


def loaddata(file):
    data = np.loadtxt(file, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?'), 94: lambda x:int(x[6])-1 } )
    data = data[1:,]
    return data

tsnesaved = True
xgbsaved = False

thisdir = '/home/jr/Documents/Kaggle/Otto Group/TrainingData/'
thisdir2 = '/home/jr/Documents/Kaggle/Otto Group/TestData/'
trainfile = thisdir + 'train.csv'
testfile = thisdir2 + 'test.csv'
tnsefile = thisdir + 'tsne.csv'
xgbfile = thisdir + 'xgb.csv'

train = np.loadtxt(trainfile, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?'), 94: lambda x:int(x[6])-1 } )[:,1:]
test = np.loadtxt(testfile, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?') } )[:,1:]

y = train[:,-1]
x = np.concatenate((train[:,:-1],test), axis = 0)

x = 1/(1+np.exp(-np.sqrt(x)))

if not tsnesaved:
    mytsne = tsne.bh_sne(x)
    np.savetxt(tnsefile, mytsne, delimiter=',', newline='\n')
if tsnesaved:
    mytnse = np.loadtxt(tnsefile, delimiter=',')

if not xgbsaved:
    param = {}
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    param['num_class'] = 9
    param['max_depth'] = 8
    param['eta'] = .03
    param['min_child_weight'] = 3
    param['silent'] = 1
    num_round = 1200

    np.round(x, 3)
    x = np.append(x, mytnse, axis=1)

    trainlen = y.shape[0]

    trainX = x[:(trainlen),:]
    testX = x[(trainlen):,:]

    xgtrain = xgb.DMatrix(trainX, label=y)
    xgtest = xgb.DMatrix(testX)
    watchlist = [(xgtrain, 'train')]
    bst = xgb.train(param, xgtrain, num_round, watchlist)

    pred = bst.predict(xgtest)
    np.savetxt(xgbfile, pred, delimiter=',', newline='\n')

if xgbsaved:
    pred = np.loadtxt(xgbfile, delimiter=',')