import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

data = np.load('../watermelon_3.0a.npz')

x = data['arr_0'][:, 0].reshape([-1, 1])  # n * d
y = data['arr_0'][:, 1]  # 1 * n

clf = svm.SVR()
clf.fit(x, y)

res = clf.predict([[0.5]])
print(res)
