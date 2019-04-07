import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = np.load('../watermelon_3.0a.npz')

x = data['arr_0']  # n * d
y = data['arr_1']  # 1 * n

clf = svm.SVC(kernel='rbf', random_state=0, gamma=0.80, C=100.0)
clf.fit(x, y)

x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))

z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

z = z.reshape(xx1.shape)

colors = ('red', 'blue', 'lightgreen', 'cyan','gray')
cmap = ListedColormap(colors[:len(np.unique(y))])
markers = ('s', 'x', 'o', '^', 'v')

plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

plt.show()

