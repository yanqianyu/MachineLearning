import numpy as np
from math import log
import operator


def calcEnt(dataSet):
    dataSet = np.array(dataSet)
    num = len(dataSet)
   # type_num = np.unique(dataSet[:, -1])
    typeCounts = {}
    for type in dataSet[:, -1]:
        if type not in typeCounts.keys():
            typeCounts[type] = 0
        typeCounts[type] += 1
    shannonEnt = 0.0
    for key in typeCounts:
        prob = float(typeCounts[key])/num
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


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
                new_dataSet.append(np.hstack((data[:axis], data[axis+1:])))
        else:
            if data[axis] <= value:
                new_dataSet.append(np.hstack((data[:axis], data[axis + 1:])))
    return new_dataSet


def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEnt = calcEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # get all value about feature[i]
        featList = [data[i] for data in dataSet]
        # discrete
        if i in range(xn_discrete):
            newEnt = 0.0
            # get split point
            uniqueVals = np.unique(featList)
            for value in uniqueVals:
                subDataset = splitDataDiscrete(dataSet, i, value)
                prob = len(subDataset) / float(len(dataSet))
                newEnt += prob * calcEnt(subDataset)
            infoGain = baseEnt - newEnt
        # continous
        else:
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitEnt = 10000
            for value in splitList:
                newEnt = 0.0
                subDataset0 = splitDataContinous(dataSet, i, value, 0)
                subDataset1 = splitDataContinous(dataSet, i, value, 1)

                prob0 = len(subDataset0) / float(len(dataSet))
                newEnt += prob0 * calcEnt(subDataset0)
                prob1 = len(subDataset1) / float(len(dataSet))
                newEnt += prob1 * calcEnt(subDataset1)
                if newEnt < bestSplitEnt:
                    bestSplitEnt = newEnt
                    bestSplitValue = value
            infoGain = baseEnt - bestSplitEnt

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    if bestFeature > xn_discrete:
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
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

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

    for value in uniqueVals:
        myTree[bestFeat][value] = createTree(splitDataDiscrete(dataSetPart, bestFeat, value), dataSetFull)

    return myTree

if __name__=='__main__':
    data = np.load('../watermelon_3.0.npz')
    xn_discrete = data['arr_0']
    xn_continuous = data['arr_1']
    yn = data['arr_2']
    x_discrete = data['arr_3']
    x = data['arr_4']  # n * d
    y = data['arr_5']  # 1 * n
    y = np.reshape(y, [17, 1])
    dataSet = np.hstack((x, y))

    myTree = createTree(dataSet, dataSet)
    print(myTree)