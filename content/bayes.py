# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:32:19 2018
朴素贝叶斯
@author: Administrator
"""

#创建数据集,返回词条切分后的文档集合(postingList)和类别标签
def loadDataSet():
    postingList = [["my","dog","has","flea",\
                   "problem","help","please"],
                    ["maybe","not","take","him",\
                     "to","dog","park","stupid"],
                     ["my","dalmation","is","so","cute",\
                      "I","love","him"],
                      ["stop","posting","stupid","worthless","garbage"],
                      ["mr","licks","ate","my","steak","how",\
                       "to","stop","him"],
                       ["quit","buying","worthless","dog","food","stupid"]]
    #1代表侮辱性文字，0代表正常言论
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

#返回所有文档中不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#输入为词汇表(vocabList)及某个文档(inputSet),对文档中出现的每个词在词汇表中做相应的标记(接下)
#(接上)出现就标记为1，没出现就标记为0，返回最后的标记结果
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:print "the word: %s is not in my vocabulary!" % word
    return returnVec

#朴素贝叶斯分类器训练函数
"""
trainMatrix:经过标记处理的向量矩阵
trainCategory:类别矩阵
return:在每种类别中各个词出现的概率及是侮辱性言论的概率
"""
def trainNB0(trainMatrix,trainCategory):
    #言论条数
    numTrainDocs = len(trainMatrix)
    #去重后的词数
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    #初始化每个类别的向量
    p0Num = np.zeros(numWords);p1Num = np.zeros(numWords)
    #保存每种类别的词语数字
    p0Denom = 0.0;p1Denom = 0.0
    #遍历每一条言论，按照言论的类别与相应的向量做向量加法，对每种类别中词语出现的次数做统计
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #计算每种类别中各单词出现的概率
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect,p1Vect,pAbusive
