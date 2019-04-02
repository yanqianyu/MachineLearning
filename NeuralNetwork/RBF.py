import numpy as np
from numpy.linalg import norm
import math


def RBF(x, beta, c):
    return norm(x - c, 2)


def RBF_BP(src, trg, hiddenLayer, lr):
    error = 0.001
    w = np.random.rand(hiddenLayer)
    b = np.random.rand(hiddenLayer)
    beta = np.random.rand(hiddenLayer)
    center = np.random.rand(hiddenLayer)
    trainNum = 0
    flag = 1
    while flag:
        trainNum += 1
        for k in range(len(src)):
            x = src[k]
            y = trg[k]

            yt = 0.0
            for h in range(hiddenLayer):
                b[h] = RBF(x, beta[h], center[h])
                yt += w[h] * b[h]

            g = np.zeros(hiddenLayer)  # grad
            for h in range(hiddenLayer):
                g[h] = (yt - y) * b[h]

            for h in range(hiddenLayer):
                beta[h] += lr * g[h] * w[h] * norm(x - center[h], 2)
                w[h] -= lr * g[h]

            loss = (yt - y) ** 2

            print(trainNum, ' ', loss)
            if loss < error:
                flag = 0
                break

    y_pre = []
    for k in range(len(src)):
        x = src[k]

        yt = 0.0
        for h in range(hiddenLayer):
            b[h] = RBF(x, beta[h], center[h])
            yt += w[h] * b[h]
        y_pre.append(yt)

    print('predict ', y_pre)


if __name__=='__main__':
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
    )

    y = np.array(
        [
            [0],
            [1],
            [1],
            [0]
        ]
    )

    RBF_BP(x, y, 5, 0.05)

