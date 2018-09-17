# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 23:21:34 2018
决策树
@author: Administrator
"""

from math import log
#计算并返回香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#创建测试数据集
def createDataSet():
    dataSet = [[1,1,'maybe'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ["no surfacing","flippers"]
    return dataSet,labels

#划分数据集，dataSet：数据集 axis：特征标号 value：特征值
#返回value为axis的数据集(刨除axis号特征)
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    #遍历数据集，将axis号特征为value的数据项处理后返回
    for featVec in dataSet:
        if featVec[axis] == value:
           reducedFeatVec = featVec[:axis]
           reducedFeatVec.extend(featVec[axis + 1:])
           retDataSet.append(reducedFeatVec)
    return retDataSet

#选取最适合分割的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
