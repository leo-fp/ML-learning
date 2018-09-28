# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:24:32 2018
Logistic回归
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
