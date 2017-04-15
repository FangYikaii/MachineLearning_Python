#-*- coding:utf-8 -*-
from pylab import *
from numpy import *

def pca(data,nRedDim=0,normalise=1):
    # 数据标准化
    m = mean(data,axis=0)
    data -= m
    # 协方差矩阵
    C = cov(transpose(data))
    # 计算特征值特征向量，按降序排序
    evals,evecs = linalg.eig(C)
    indices = argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    if nRedDim>0:
        evecs = evecs[:,:nRedDim]

    if normalise:
        for i in range(shape(evecs)[1]):
            evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
    # 产生新的数据矩阵
    x = dot(transpose(evecs),transpose(data))
    # 重新计算原数据
    y=transpose(dot(evecs,x))+m
    return x,y,evals,evecs

x = random.normal(5,.5,1000)
y = random.normal(3,1,1000)

a = x*cos(pi/4) + y*sin(pi/4)
b = -x*sin(pi/4) + y*cos(pi/4)

plot(a,b,'.')

xlabel('x')
ylabel('y')

title('raw dataset')

data = zeros((1000,2))
data[:,0] = a
data[:,1] = b
x,y,evals,evecs = pca(data,1)
print(y)
figure()
plot(y[:,0],y[:,1],'.')
xlabel('x')
ylabel('y')
title('new dataset')
show()