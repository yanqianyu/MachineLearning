import numpy as np
from math import log
import operator

def logistic():
    pass


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