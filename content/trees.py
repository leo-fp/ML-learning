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

#计算每个特征分割后的熵，返回最佳特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)   #原始熵
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

#返回类别列表中出现次数最多的值
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet,labels):
    #保存所有类别
    classList = [example[-1] for example in dataSet]
    #当前数据集中所有类型相同,直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #用完所有特征后，无法得到同意的类别，调用majorityCnt返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #bestFeat:最佳分割特征编号
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #创建当前数据集的字典
    myTree = {bestFeatLabel:{}}
    #删除类别列表中对最佳特征的引用
    #这一句会使特征标签的数量减少，在调用classify时会出现not in list错误
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #消除labels[bestFeat]后复制给subLabels
        subLabels = labels[:]
        #递归创建树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#分类
"""
inputTree:创建好的树
featValues:特征标签
testVec:输入样例

return:预测类别

ps:调动该函数时确保featValues的数量正确
"""
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=="dict":
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel

#决策树的创建很耗时，可以将决策树存在磁盘上，使用时直接从磁盘上读取
#将决策树存入磁盘
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,"w")
    pickle.dump(inputTree,fw)
    fw.close()
    
#从磁盘取出决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
