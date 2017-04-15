# -*- coding: utf-8 -*-
#################################################
# kmeans: k-means cluster
# Author : FangYikai
# Date   : 2016-09-15
#################################################
from numpy import *
import time
import matplotlib.pyplot as plt
import kmeans
import xlrd
from numpy import *

## step 1: load data
print "step 1: load data..."
wb = xlrd.open_workbook('Feature.xls')  # 读取xls
sh = wb.sheet_by_index(0)  # 读取第一个表格
dataSet = []  # 定义数据存储
for i in range(0, 3):
    dataSet.append(sh.col_values(i))  # list 添加
dataSet = map(list, zip(*dataSet)) # 转置
dataSet.pop()  # 删一列

## step 2: clustering...
print "step 2: clustering..."
dataSet = mat(dataSet)  # list转mat
k = 2
centroids, clusterAssment = kmeans.kmeans(dataSet, k)

## step 3: show the result
print "step 3: show the result..."
numSamples, dim = dataSet.shape
if dim != 3:
    print "Sorry! I can not draw because the dimension of your data is not 2!"

mark = ['y', 'r', 'g', 'k']
if k > len(mark):
    print "Sorry! Your k is too large!"

fig = plt.figure()    # 定义一个图
ax3D = fig.add_subplot(111, projection='3d')  # 3d
# draw all samples
for i in xrange(numSamples):
    markIndex = int(clusterAssment[i, 0])       # 画点
    ax3D.scatter(dataSet[i, 0], dataSet[i, 1], dataSet[i, 2],c=mark[markIndex],s=100)

# draw the centroids
for i in range(k):
    ax3D.scatter(centroids[i, 0], centroids[i, 1], centroids[i, 2], c=mark[i],s=500)  # 中心点

ax3D.set_zlabel('Z')
ax3D.set_ylabel('Y')
ax3D.set_xlabel('X')
plt.show()