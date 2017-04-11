#/etc/usr/bin python3
#-*- coding:utf-8 -*-

#a practice for k-nearst neighbors
from numpy import *
import operator

#collect data
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']

    return group,labels


#prepare 
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())   # lines in file
    returnMat = zeros((numberOfLines,3))    # numpy.zeros(shape,dtype=float,order='C') Return a new array of given shape and type,filled with zeros
    classLabelVector = []
    fr = open(filename)                  #? why open the file again
    index = 0
    for line in fr.readlines():
        line = line.strip()               # returns a copy of the string in which all chars hava been stripped from the beginning and end of the string (default whitespace characters)
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]   # matrix  index row, :
        classLabelVector.append(listFromLine[-1])  #? why int(listFromLine[-1]) in book?
        index+=1
    return returnMat,classLabelVector


#analyze
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]          #length of first dimension
    diffMat = tile(inX,(dataSetSize,1))-dataSet # numpy.tile repeat inX 
    sqDiffMat = diffMat**2                      #square
    sqDistances = sqDiffMat.sum(axis=1)         #can't clearly describe it, means sum by row
    distances = sqDistances**0.5                #square root
    sortedDistIndices = distances.argsort()     #sorted by square root distance
    classCount = {}                             # count the k nearest neighbors 
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1  #get(..,0)  if get null return 0
    sortedClassCount =[(k,classCount[k]) for k in sorted(classCount,key=classCount.get,reverse=True)] #
    return sortedClassCount[0][0]

#group,labels = createDataSet()

#print(classify0([0,0],group,labels,3))
