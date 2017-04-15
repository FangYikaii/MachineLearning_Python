# -*- coding: utf-8 -*-
from matplotlib import pyplot
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import xlrd
from numpy import *

dataSet = []
labels = []
fileIn = open('testSet.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])
    labels.append(float(lineArr[2]))
for index,m in enumerate(labels):
    if m==-1.0:
        labels[index]=0.0
dataSet = mat(dataSet)
labels = mat(labels).T
x = dataSet[0:101, :]
y = labels[0:101, :]
x_train = dataSet[0:81, :]
y_train = labels[0:81, :]
x_test = dataSet[80:101, :]
y_test = labels[80:101, :]



# 调用MultinomialNB分类器
clf = MultinomialNB().fit(x_train, y_train)
doc_class_predicted = clf.predict(x_test)

# print(doc_class_predicted)
# print(y)
print(np.mean(doc_class_predicted == y_test))

# 准确率与召回率
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
answer = clf.predict_proba(x_test)[:, 1]
report = answer > 0.5
print(classification_report(y_test, report, target_names=['neg', 'pos']))