# linear discriminant Analysis
import numpy as np
import matplotlib.pyplot as plt
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

x0 = np.array(x[:8])
x1 = np.array(x[8:])

# positive mean
miu0 = np.mean(x0, axis=0).reshape((-1, 1))
# negative mean
miu1 = np.mean(x1, axis=0).reshape((-1, 1))

cov0 = np.cov(x0, rowvar=False)
cov1 = np.cov(x1, rowvar=False)

S_w = np.mat(cov0 + cov1)
w = S_w.I * (miu0 - miu1)

plt.scatter(x0[:, 0], x0[:, 1], c='b', label='+', marker='+')
plt.scatter(x1[:, 0], x1[:, 1], c='r', label='-', marker='_')

plt.plot([0, 1], [0, -w[0]/w[1]], label='y')
plt.xlabel('密度', fontsize=15, color='green')
plt.ylabel('含糖率', fontsize=15, color='green')

plt.legend()
plt.show()