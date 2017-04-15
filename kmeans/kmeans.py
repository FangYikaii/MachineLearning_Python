# -*- coding: utf-8 -*-
#################################################
# kmeans: k-means cluster
# Author : FangYikai
# Date   : 2016-09-15
#################################################
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim)) # 分类数为行，x的特征种类为列
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to, # 第一列为中心
    # second column stores the error between this sample and its centroid # 第二列误差
    clusterAssment = mat(zeros((numSamples, 2)))  #定义（数据长度，2）的数组
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)  # 随机初始化中心点

    while clusterChanged:  #迭代
        clusterChanged = False
        ## for each sample
        for i in xrange(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])   # 计算与各个点的距离
                if distance < minDist:
                    minDist = distance
                    minIndex = j  # 判断，看是属于哪类

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print 'Congratulations, cluster complete!'
    return centroids, clusterAssment    # 中心点【k个特征的中心点】  和 分类标【对应的类别，以及力中心点的距离】


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print "Sorry! Your k is too large"
        return 1

    # draw all samples
    for i in xrange(numSamples): # 显示二维图的散点
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])   #

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)  # 显示中心点

    plt.show()