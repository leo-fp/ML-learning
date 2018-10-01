# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:24:32 2018
Logistic回归:
   不断的调整Sigmoid函数的最佳拟合参数
@author: Administrator
"""
import math
import numpy as np

#读取文本文件并格式化
#return:m * 3的矩阵(不包括类别),和m * 1的类别矩阵
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#Sigmoid函数，值域为(0,1)
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

#梯度上升算法:将输入矩阵(mxn)与系数矩阵(nx1)做矩阵乘法
#得到(mx1)的矩阵，将该矩阵作为Sigmoid函数的输入，将结果
#调整到(0,1),之后计算实际类别(labelMat)与预测类别(h)的误差(error)
#之后用误差调整回归系数
#dataMatIn:输入矩阵(mxn)
#classLabels:类别矩阵(mx1)
#return:回归系数(nx1)
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    #步长
    alpha = 0.001
    #迭代次数
    maxCycles = 500
    #回归系数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(np.dot(dataMatrix,weights))
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

#画出数据集和Logistic回归最佳拟合直线
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = "red",marker = "s")
    ax.scatter(xcord2,ycord2,s = 30,c = "green")
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.xlabel("x1");plt.ylabel("x2");
    plt.show()

#改进的随机梯度上身算法
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter): 
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0,len(dataMatrix)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
    return weights

#Logistic回归分类函数
#用调整好的参数与输入数据相乘，大于0.5预测类别为1，否则为0
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0
