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
