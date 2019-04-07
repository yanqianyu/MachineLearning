import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

data = np.load('../watermelon_3.0a.npz')

x = data['arr_0']  # n * d
y = data['arr_1']  # 1 * n

clf = svm.SVC(kernel='linear', C=100.0)
clf.fit(x, y)

w = clf.coef_[0]  # 获取w
a = -w[0] / w[1]  # 斜率
# 画图划线
xx = np.linspace(-5, 5)  # (-5,5)之间x的值
yy = a * xx - (clf.intercept_[0]) / w[1]  # xx带入y，截距

# 画出与点相切的线
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print("W:", w)
print("a:", a)

print("support_vectors_:", clf.support_vectors_)
print("clf.coef_:", clf.coef_)

plt.figure(figsize=(8, 4))
plt.plot(xx, yy)
plt.plot(xx, yy_down)
plt.plot(xx, yy_up)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)  # [:，0]列切片，第0列

plt.axis('tight')

plt.show()





