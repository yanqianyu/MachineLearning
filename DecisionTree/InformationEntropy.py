import numpy as np
import pandas as pd

data = np.load('../watermelon_3.0.npz')


def calcent(x, y):
    num = len(x)
    type_num = np.unique(y)
    typeCounts = {}
    for type in y:
        typeCounts[type] += 1
    shannonEnt = 0.0
    for key in typeCounts:
        prob = float(typeCounts[key])/num
        shannonEnt -= prob * np.log(prob, 2)
    return shannonEnt


def splitDataDiscrete(x, y, axis, value):
    new_x = []
    new_y = []
    for data in x:
        if data[axis] == value:
            new_x.append(x[:axis] + data[axis+1:])
            new_y.append()

    return new_x, new_y


def splitDataContinous(x, y, axis, value, direction):
    new_x = []
    new_y = []
    for data in x:
        if direction == 0:
            if data[axis] > value:
                new_x.append(data[:axis] + data[axis+1:])
        else:
            if data[axis] <= value:
                new_x.append(data[:axis] + data[axis + 1:])
    return new_x


def chooseBestFeature(x, y):
    numFeatures = len(x[0]) - 1
    baseEnt = calcent(x, y)
    bestInfoGain = 0.0
    beatFeature = -1
    bestSplitData = []
    for i in range(numFeatures):
        featList = [data[i] for data in x]
        # discrete

        # continous