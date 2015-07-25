__author__ = 'jr'

import numpy as np
import xgboost as xgb
import tsne
from sklearn.ensemble import RandomForestClassifier

def loaddata(file):
    data = np.loadtxt(file, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?'), 94: lambda x:int(x[6])-1 } )
    data = data[1:,]
    return data

tsnesaved = True
xgbsaved = True

thisdir = '/home/jr/Documents/Kaggle/Otto Group/TrainingData/'
thisdir2 = '/home/jr/Documents/Kaggle/Otto Group/TestData/'
trainfile = thisdir + 'train.csv'
testfile = thisdir2 + 'test.csv'
tnsefile = thisdir + 'tsne.csv'
xgbfile = thisdir + 'xgb.csv'
predfile = thisdir2 + 'pred.csv'
counterfile = thisdir + 'counter.csv'

train = np.loadtxt(trainfile, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?'), 94: lambda x:int(x[6])-1 } )[:,1:]
test = np.loadtxt(testfile, delimiter=',', skiprows=1, converters={0: lambda x:int(x == '?') } )[:,1:]
counter = np.loadtxt(counterfile, delimiter=',').reshape([1])[0]

y = train[:,-1]
x = np.concatenate((train[:,:-1],test), axis = 0)

x = 1/(1+np.exp(-np.sqrt(x)))

if not tsnesaved:
    mytsne = tsne.bh_sne(x)
    np.savetxt(tnsefile, mytsne, delimiter=',', newline='\n')
if tsnesaved:
    mytnse = np.loadtxt(tnsefile, delimiter=',')

np.round(x, 3)
x = np.append(x, mytnse, axis=1)
trainlen = y.shape[0]
trainX = x[:(trainlen),:]
testX = x[(trainlen):,:]

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

    xgtrain = xgb.DMatrix(trainX, label=y)
    xgtest = xgb.DMatrix(testX)
    watchlist = [(xgtrain, 'train')]
    bst = xgb.train(param, xgtrain, num_round, watchlist)

    pred = bst.predict(xgtest)
    np.savetxt(xgbfile, pred, delimiter=',', newline='\n')

if xgbsaved:
    pred = np.loadtxt(xgbfile, delimiter=',')

tmpL = trainlen
trind = np.arange(trainlen)
gtree = 200
nloops = 240

if counter > 0:
    pred = np.loadtxt(predfile, delimiter=',')

for z in range(counter, nloops):
    print z
    tmpS1 = np.random.choice(trind, size=tmpL, replace=True)
    tmpS2 = np.setdiff1d(trind, tmpS1)

    tmpX2 = trainX[tmpS2]
    tmpY2 = y[tmpS2]

    clf = RandomForestClassifier(n_estimators=100, n_jobs=4)
    clf.fit(tmpX2, tmpY2)

    tmpX1 = trainX[tmpS1]
    tmpY1 = y[tmpS1]

    tmpX2 = clf.predict(tmpX1).reshape(-1,1)
    tmpX3 = clf.predict(testX).reshape(-1,1)

    param = {}
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    param['num_class'] = 9
    param['subsample'] = 0.8
    param['max_depth'] = 11
    param['eta'] = .46
    param['min_child_weight'] = 10
    param['silent'] = 1
    num_round = 60

    newtrainx = np.concatenate((tmpX1,tmpX2), axis = 1)
    xgtrain = xgb.DMatrix(newtrainx, label=tmpY1)

    newtestx = np.concatenate((testX,tmpX3), axis = 1)
    xgtest = xgb.DMatrix(newtestx)
    watchlist = [(xgtrain, 'train')]

    bst = xgb.train(param, xgtrain, num_round, watchlist)
    pred0 = bst.predict(xgtest)
    pred = pred + pred0
    np.savetxt(counterfile, z, delimiter=',', newline='\n')
    np.savetxt(predfile, pred, delimiter=',', newline='\n')
pred = pred / (nloops+1)
np.savetxt(predfile, pred, delimiter=',', newline='\n')
