#--*-- coding: utf-8--*--
"""
k-近邻算法:
	将输入实例的特征向量和类别已定的训练数据进行比对，取最
	相似的k个，在这k个数据中，出现次数最多的类别为该实例的
	预测类别
"""
from numpy import *
import operator

#创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group,lables

#inX:代预测实例，dataSet:特征向量集，lables:类别集合
def classify(inX,dataSet,lables,k):
    #距离度量 量度公式为欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #从小到大排序，返回索引值
    sortedDisIndicies = distances.argsort()
    #循环k次，将前k项数据的类别及出现的次数存储在字典classCount中
    classCount = {}
    for i in range(k):
        voteIlable = lables[sortedDisIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0) + 1	
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
