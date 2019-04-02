import numpy as np
import math


def get_data():
    data = np.load('../watermelon_3.0.npz')

    xn_discrete = data['arr_0']
    xn_continuous = data['arr_1']
    yn = data['arr_2']
    x_discrete = data['arr_3']
    x = data['arr_4']  # n * d
    y = data['arr_5']  # 1 * n
    y = np.reshape(y, [17, 1])  # n * 1

    return x, y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def abp(hiddenLayer):
    '''
    :param hiddenLayer: the number of hidden layers
    :return:
    '''
    x, y = get_data()  # x [n, d]  y [n, 1]
    v = np.random.rand(x.shape[1], hiddenLayer)  # [d, h]
    w = np.random.rand(hiddenLayer, y.shape[1])  # [h, l]

    t0 = np.random.rand(1, hiddenLayer)  # [1, h]
    t1 = np.random.rand(1, y.shape[1])  # [1, l]

    learning_rate = 0.1
    maxTrainNum = 100000
    trainNum = 0
    loss = 0
    error = 0.001
    flag = 1

    while flag:
        b = sigmoid(x.dot(v) - t0)  # [n, h]
        y0 = sigmoid(b.dot(w) - t1)  # [l, 1]
        loss = sum((y - y0) ** 2) / x.shape[0]
        if loss < error or trainNum > maxTrainNum:
            break
        trainNum += 1
        g = y0 * (1 - y0) * (y - y0)  # [n, l]
        e = b * (1 - b) * g.dot(w.T)  # [n, h]
        w += learning_rate * b.T.dot(g)
        t1 -= learning_rate * g.sum(axis=0)
        v += learning_rate * x.T.dot(e)
        t0 -= learning_rate * e.sum(axis=0)

        print('trainNum: ', trainNum, ' loss: ',loss)

    print('trainNum = ', trainNum)

if __name__=='__main__':
    abp(5)