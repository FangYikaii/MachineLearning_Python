# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
''' 数据生成 '''
h = 0.1
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
n = xx.shape[0]*xx.shape[1]
x = np.array([xx.T.reshape(n).T, xx.reshape(n)]).T
y = (x[:,0]*x[:,0] + x[:,1]*x[:,1] < 0.8)
y.reshape(xx.shape)

x_train, x_test, y_train, y_test\
    = train_test_split(x, y, test_size = 0.2)

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
 #   print(answer)
 #   print(y_train)

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Put the result into a color plot
    z = answer.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlabel(u'X')
    plt.ylabel(u'Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
