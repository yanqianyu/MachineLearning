import numpy as np
import pandas as pd
x = np.array([
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.328],
    [0.556, 0.215],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.437, 0.211],
    [0.666, 0.091],
    [0.243, 0.267],
    [0.245, 0.057],
    [0.343, 0.099],
    [0.639, 0.161],
    [0.657, 0.198],
    [0.360, 0.370],
    [0.593, 0.042],
    [0.719, 0.103]
])

x = list(x.T)
x = np.array([x[0],x[1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])  # 3 * 17
y = np.array([1, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0])

beta = np.array([[0],[0],[1]])
l_old = 0
n = 0

while 1:
    l_cur = 0
    beta_T_x = np.dot(beta.T[0], x)

    for i in range(17):  # the num of datas is 17
        l_cur = l_cur + (-y[i]*beta_T_x[i] + np.log(1 + np.exp(beta_T_x[i])))

    if np.abs(l_cur - l_old) <= 0.0000001:
        break

    n = n+1
    l_old = l_cur
    dbeta = 0  # first order
    dbeta2 = 0  # second order
    for i in range(17):
        dbeta = dbeta - np.dot(np.array([x[:, i]]).T,
                               (y[i] - (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i])))))
        dbeta2 = dbeta2 + np.dot(np.array([x[:, i]]).T, np.array([x[:, i]]).T.T) \
                 * (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i]))) \
                 * (1 - (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i]))))
    beta = beta - np.dot(np.linalg.inv(dbeta2), dbeta)
print(beta)
