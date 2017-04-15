#-*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def loadDataSet(fileName):
    # 读取EXCEL中内容到数据库中
    wb = xlrd.open_workbook(fileName)
    sh = wb.sheet_by_index(0)  # 第一个表
    dfun = []
    for i in range(0, 4):
        dfun.append(sh.col_values(i))
    x = map(list, zip(*dfun))
    x.pop()
    x = mat(x)
    return mat(x)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def plotBestFit(dataSet1, dataSet2):
    dataArr1 = array(dataSet1)
    dataArr2 = array(dataSet2)
    n = shape(dataArr1)[0]
    n1 = shape(dataArr2)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    xcord3 = [];
    ycord3 = []
    j = 0
    for i in range(n):
        xcord1.append(dataArr1[i, 0]);
        ycord1.append(dataArr1[i, 1])
        xcord2.append(dataArr2[i, 0]);
        ycord2.append(dataArr2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


if __name__ == '__main__':
    mata = loadDataSet('Feature.xls')
    a, b = pca(mata, 2)
    plt.scatter(a[:,0],a[:,1])
    plt.show()
    print(a)

