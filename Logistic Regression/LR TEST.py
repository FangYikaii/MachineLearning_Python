# -*- coding: utf-8 -*-
#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################
from sklearn.cross_validation import train_test_split
from numpy import *
import matplotlib.pyplot as plt
import time
from logRegression import*
import xlrd
def loadData():
    # 读取EXCEL中内容到数据库中
    wb = xlrd.open_workbook('Feature.xls')
    sh = wb.sheet_by_index(0)
    dfun = []
    nrows = sh.nrows  # 行数
    ncols = sh.ncols  # 列数
    fo = []
    dfun.append(sh.col_values(4))
    for i in range(0, 4):
        dfun.append(sh.col_values(i))
    x = map(list, zip(*dfun))
    x.pop()
    fo.append(sh.col_values(4))
    y = fo
    y = map(list, zip(*y))
    y.pop()
    train_x, x_test, train_y, y_test = train_test_split(x, y, test_size=0.0)
    y = []
    for j in range(0, nrows-1):
        y.append(train_y[j][0])
    train_y=y

    return mat(train_x), mat(train_y).transpose()


# step 1: load data
print ('step 1: load data...')
train_x, train_y = loadData()
test_x = train_x; test_y = train_y

## step 2: training...
print ('step 2: training...')
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights = trainLogRegres(train_x, train_y, opts)

## step 3: testing
print ('step 3: testing...')
accuracy = testLogRegres(optimalWeights, test_x, test_y)

## step 4: show the result
print ('step 4: show the result...')
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))
L=[]
for j in range(0, 125):
    x=train_x[j, 1]
    L.append((train_x[j,0],train_x[j,3],train_x[j,4]))
train_x = L
train_x=mat(train_x)
showLogRegres(optimalWeights, train_x, train_y)