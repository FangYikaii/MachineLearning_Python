# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import scipy as sp
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfVectorizer
import xlrd

# 读取EXCEL中内容到数据库中
wb = xlrd.open_workbook('Feature.xls')
sh = wb.sheet_by_index(0)  # 第一个表
dfun = []
nrows = sh.nrows  # 行数
ncols = sh.ncols  # 列数
fo = []
for i in range(1, 2):
    dfun.append(sh.col_values(i))
for i in range(3, 4):
    dfun.append(sh.col_values(i))
x=map(list, zip(*dfun))
x.pop()

fo.append(sh.col_values(6))
y=fo
y=map(list, zip(*y))
y.pop()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

x_train=mat(x_train)
y_train=mat(y_train)
x_test=mat(x_test)
y_test=mat(y_test)  # list 转 mat

h = .02
# create a mesh to plot in
x_min, x_max = x_train[:, 0].min() - 0.1, x_train[:, 0].max() + 0.1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


''' SVM '''
# title for the plots
titles = ['LinearSVC (linear kernel)',
          'SVC with polynomial (degree 3) kernel',
          'SVC with RBF kernel',
          'SVC with Sigmoid kernel']
clf_linear = svm.SVC(kernel='linear').fit(x, y)
# clf_linear  = svm.LinearSVC().fit(x, y)
clf_poly = svm.SVC(kernel='poly', degree=3).fit(x, y)
clf_rbf = svm.SVC().fit(x, y)
clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x, y)

for i, clf in enumerate((clf_linear, clf_poly, clf_rbf, clf_sigmoid)):
    answer = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print(clf)
    print(np.mean(answer == y_train))
  #  print(answer)
 #   print(y_train)

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Put the result into a color plot
    z = answer.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    # Plot also the training points
    tt = y_train.tolist()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=tt, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.xlabel(u'x')
    plt.ylabel(u'y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()