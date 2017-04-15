# -*- coding: utf-8 -*-
#################################################
# SVM: support vector machine
# Author : FangYikaii
# Date   : 2016-9-16
#################################################
from numpy import *
import numpy as np
import SVM
from sklearn.cross_validation import train_test_split
################## test svm #####################
## step 1: load data
print "step 1: load data..."
h = 0.1
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
n = xx.shape[0]*xx.shape[1]
x = np.array([xx.T.reshape(n).T, xx.reshape(n)]).T
y = (x[:,0]*x[:,0] + x[:,1]*x[:,1] < 0.8)
y.reshape(xx.shape)
yy=[]
for index,m in enumerate(y):
	if m == True:
		yy.append(1.0)
	else:
		yy.append(-1.0)
train_x, test_x, train_y, test_y\
    = train_test_split(x, yy, test_size = 0.2)
train_x=mat(train_x)
train_y = map(list, zip(train_y)) # 转置
train_y=mat(train_y)
test_x=mat(test_x)
test_y = map(list, zip(test_y)) # 转置
test_y=mat(test_y)
## step 2: training...
print "step 2: training..."
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 0))

## step 3: testing
print "step 3: testing..."
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
SVM.showSVM(svmClassifier)