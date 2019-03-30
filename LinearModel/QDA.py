import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

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

x = np.array(x)  # 17 * 2

N = x.shape[0]
d = x.shape[1]

y = np.array([1, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0])
y = y.reshape(y.size)

classes = np.unique(y)
means = np.zeros((d, classes.size))
cov = [np.zeros((d,d))] * classes.size

x0 = np.array(x[:8])
x1 = np.array(x[8:])

means[:,0] = np.mean(x0,0)
cov[0] = np.cov(x0.T)
means[:,1] = np.mean(x1,0)
cov[1] = np.cov(x1.T)

colors = iter(cm.rainbow(np.linspace(0,1,classes.size)))
plt.clf()
ax = plt.subplot(111, aspect='equal')

for i in range(classes.size):
    co = next(colors)
    lambda_, v = np.linalg.eig(cov[i])
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(means[0, i - 1], means[1, i - 1]),
                  width=lambda_[0] * 2, height=lambda_[1] * 2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), color='black')
    ell.set_facecolor('none')
    ax.add_artist(ell)

plt.scatter(x[:8, 0], x[:8, 1], c='b', label='+', marker='+')
plt.scatter(x[8:, 0], x[8:, 1], c='r', label='-', marker='_')

# mean point
plt.scatter(means[0,0], means[1,0], c='black', s=30)
plt.scatter(means[0,1], means[1,1], c='black', s=30)

plt.show()