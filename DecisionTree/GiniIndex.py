import numpy as np
from math import log
import operator

'''
Todo:
post
pre
'''
def calcGini(dataSet):
    dataSet = np.array(dataSet)
    num = len(dataSet)
    # type_num = np.unique(dataSet[:, -1])
    typeCounts = {}
    for type in dataSet[:, -1]:
        if type not in typeCounts.keys():
            typeCounts[type] = 0
        typeCounts[type] += 1
    Gini = 1.0
    for key in typeCounts:
        prob = float(typeCounts[key])/num
        Gini -= prob * prob
    return Gini


def splitDataDiscrete(dataSet, axis, value):
    new_dataSet = []
    for data in dataSet:
        # data = np.reshape(data, [1, len(data)])
        if data[axis] == value:
            new_dataSet.append(np.hstack((data[:axis], data[axis+1:])))
    return np.array(new_dataSet)


def splitDataContinous(dataSet, axis, value, direction):
    new_dataSet = []
    for data in dataSet:
        if direction == 0:
            if data[axis] > value:
                new_dataSet.append(data)
                # new_dataSet.append(np.hstack((data[:axis], data[axis+1:])))
        else:
            if data[axis] <= value:
                new_dataSet.append(data)
                # new_dataSet.append(np.hstack((data[:axis], data[axis + 1:])))
    return new_dataSet


def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestGini = 10000.0
    bestFeature = -1
    bestSpiltdict = {}
    for i in range(numFeatures):
        # get all value about feature[i]
        featList = [data[i] for data in dataSet]
        # discrete
        if i in range(xn_discrete):
            newGini = 0.0
            # get split point
            uniqueVals = np.unique(featList)
            for value in uniqueVals:
                subDataset = splitDataDiscrete(dataSet, i, value)
                prob = len(subDataset) / float(len(dataSet))
                newGini += prob * calcGini(subDataset)
        # continous
        else:
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitGini = 10000
            for value in splitList:
                newGini = 0.0
                subDataset0 = splitDataContinous(dataSet, i, value, 0)
                subDataset1 = splitDataContinous(dataSet, i, value, 1)

                prob0 = len(subDataset0) / float(len(dataSet))
                newGini += prob0 * calcGini(subDataset0)
                prob1 = len(subDataset1) / float(len(dataSet))
                newGini += prob1 * calcGini(subDataset1)
                if newGini < bestSplitGini:
                    bestSplitGini = newGini
                    bestSplit= value
            bestSpiltdict[i] = bestSplit
            newGini = bestSplitGini

        if newGini < bestGini:
            bestGini = newGini
            bestFeature = i

    if bestFeature > xn_discrete:
        bestSplitValue = bestSpiltdict[bestFeature]
        for i in range(len(dataSet)):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0

    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createTree(dataSetPart, dataSetFull):
    classList = [example[-1] for example in dataSetPart]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSetPart[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeature(dataSetPart)

    myTree = {bestFeat:{}}
    featVals = [example[bestFeat] for example in dataSetPart]

    uniqueVals = set(featVals)

    featValsFull = [example[bestFeat] for example in dataSetFull]
    uniqueValsFull = set(featValsFull)

    for value in uniqueVals:
        uniqueValsFull.remove(value)
        myTree[bestFeat][value] = createTree(splitDataDiscrete(dataSetPart, bestFeat, value), dataSetFull)
    for value in uniqueValsFull:
        myTree[bestFeat][value] = majorityCnt(classList)
    return myTree



if __name__=='__main__':
    data = np.load('../watermelon_2.0.npz')
    xn_discrete = data['arr_0']
    yn = data['arr_1']
    x_discrete = data['arr_2']
    x = data['arr_3']  # n * d
    y = data['arr_4']  # 1 * n
    y = np.reshape(y, [17, 1])
    dataSet = np.hstack((x, y))

    myTree = createTree(dataSet, dataSet)
    print(myTree)

# {3:
#      {0:
#           {1:
#                {0: 1,
#                 1:
#                     {0:
#                          {0:
#                               {2:
#                                    {0: 1,
#                                     1: 0,
#                                     2: 1}
#                                },
#                           1: 1,
#                           2: 1}
#                      },
#                 2: 0}
#            },
#       1:
#           {4:
#                {0: 0,
#                 1: 1,
#                 2: 0}
#            },
#       2: 0}
#  }
