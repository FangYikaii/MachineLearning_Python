# -*- coding: utf-8 -*-
# 声明：1.必须将编码注释放在第一行或者第二行 2
# .格式： 1. # coding=<encoding name>
#  格式： 2.#!/usr/bin/python
# -*- coding: <encoding name> -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

''' 数据生成 '''
x = np.arange(0, 1, 0.002)       # 【0-1范围，0.002间隔的数组】
y = norm.rvs(0, size=500, scale=0.1)  # 500个0.1维随机变量
y = y + x**2

''' 均方误差根 '''
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

''' 与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档  '''
def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()


''' 这是Conway&White《机器学习使用案例解析》里的版本 '''
def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)


plt.scatter(x, y, s=5)
degree = [1,2,10,50,80,100]
y_test = []
y_test = np.array(y_test)


for d in degree:
    clf = Pipeline([('poly', PolynomialFeatures(degree=d)),
                    ('linear', LinearRegression(fit_intercept=False))])
    clf.fit(x[:, np.newaxis], y)
    y_test = clf.predict(x[:, np.newaxis])
    print(clf.named_steps['linear'].coef_)
    print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f' %
          (rmse(y_test, y),
           R2(y_test, y),
           R22(y_test, y),
           clf.score(x[:, np.newaxis], y)))
    plt.plot(x, y_test, linewidth=2)
plt.grid()
plt.legend(['1','2','100'], loc='upper left')
plt.show()

