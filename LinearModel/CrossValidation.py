import numpy as np
import pandas as pd
import xlrd


def get_data():
    data = xlrd.open_workbook('IrisData/iris.xlsx')
    sheet = data.sheets()[0]
    x0 = []  # x * 5
    y0 = []  # 1 * x
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(1, sheet.nrows):
        if sheet.row_values(i)[-1] == 0:
            x0.append(sheet.row_values(i)[0:-1])
            y0.append(sheet.row_values(i)[-1])
        elif sheet.row_values(i)[-1] == 1:
            x1.append(sheet.row_values(i)[0:-1])
            y1.append(sheet.row_values(i)[-1])
        elif sheet.row_values(i)[-1] == 2:
            x2.append(sheet.row_values(i)[0:-1])
            y2.append(sheet.row_values(i)[-1])

    x0 = np.array(x0)
    y0 = np.array(y0).T

    x1 = np.array(x1)
    y1 = np.array(y1).T

    x2 = np.array(x2)
    y2 = np.array(y2).T

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
        p = np.diag((p1 * (1 - p1))).reshape(n)
        deta = -np.sum(x * (y - p1), 0 ,keepdims=True)

        beta -= deta * lr
        z = x.dot(beta.T)

    return beta


def test(beta, x, y):
    predicts = (x.dot(beta.T) >= 0)
    error = np.sum(predicts != y)
    return error


def holdOut(x, y):
    error = 0
    for i in range(x.shape[0]):
        train_x = x[:i] + x[i+1:]
        train_y = y[:i] + y[i+1:]
        val_x = x[i]
        val_y = y[i]
        train_x = np.array(train_x).reshape(-1, 4)
        train_y = np.array(train_y).reshape([-1, 1])
        val_x = np.array(val_x)
        val_y = np.array(val_y).reshape([-1, 1])
        beta = logistic(train_x, train_y)
        error = test(beta, val_x, val_y)