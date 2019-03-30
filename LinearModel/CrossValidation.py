import numpy as np
import pandas as pd
import xlrd


def get_data():
    data = xlrd.open_workbook('IrisData/iris.xlsx')
    sheet = data.sheets()[0]
    x0 = []  # x * 5
    y0 = []  # x * 1
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(1, sheet.nrows):
        if sheet.row_values(i)[-1] == 0:
            x0.append(sheet.row_values(i)[0:-1])
            y0.append([sheet.row_values(i)[-1]])
        elif sheet.row_values(i)[-1] == 1:
            x1.append(sheet.row_values(i)[0:-1])
            y1.append([sheet.row_values(i)[-1]])
        elif sheet.row_values(i)[-1] == 2:
            x2.append(sheet.row_values(i)[0:-1])
            y2.append([sheet.row_values(i)[-1]])

    return x0, y0, x1, y1, x2, y2


def logistic(x, y):
    # x [n, 5]
    # y [n, 1]
    n, d = x.shape
    lr = 0.001

    beta = np.ones((1, d)) * 0.1
    z = x.dot(beta.T)

    for i in range(n):
        p1 = np.exp(z) / (1 + np.exp(z))
        p = np.diag((p1 * (1 - p1))).T
        deta = -np.sum(x * (y - p1), 0, keepdims=True)

        beta -= deta * lr
        z = x.dot(beta.T)

    return beta


def test(beta, x, y):
    predicts = (x.dot(beta.T) >= 0)
    error = np.sum(predicts != y)
    return error


def tenFold(x1, x2):
    # x1 positive
    # x2 negative
    folds = []
    y = []
    n = len(x1)//10
    m = len(x2)//10
    for i in range(10):
        fold = []
        fold += x1[i*n: (i+1)*n]
        fold += x2[i*m: (i+1)*m]
        folds.append(fold)
        y.append([1]*n + [0]*m)
    return folds, y


def crossValid(folds, y):
    # folds [10, n+m, 5]
    # y [10, n+m]
    error = 0
    for i in range(10):
        train_x = folds[:i] + folds[i+1:]
        train_y = y[:i] + y[i+1:]
        val_x = folds[i]
        val_y = y[i]
        train_x = np.array(train_x).reshape(-1, 5)
        train_y = np.array(train_y).reshape([-1, 1])
        val_x = np.array(val_x)
        val_y = np.array(val_y).reshape([-1, 1])
        beta = logistic(train_x, train_y)
        error += test(beta, val_x, val_y)
    return error

if __name__=="__main__":
    x0, y0, x1, y1, x2, y2 = get_data()

    folds, y = tenFold(x0, x1)
    error1 = crossValid(folds, y)

    folds, y = tenFold(x0, x2)
    error2 = crossValid(folds, y)

    folds, y = tenFold(x1, x2)
    error3 = crossValid(folds, y)

    error = error1 + error2 + error3

    print(error)
    print(error1, ' ', error2, ' ', error3)
