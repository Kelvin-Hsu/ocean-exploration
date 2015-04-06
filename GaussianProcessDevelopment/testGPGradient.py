import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GaussianProcessModel
# from scipy.interpolate import interpn

X = np.random.rand(100, 2)
x1 = X[:, 0]
x2 = X[:, 1]
y = 2*np.exp(-((x1 - 0.5)**2 + (x2 - 0.5)**2)/0.1) # -(x1 - 0.25) ** 2 + 0 * np.random.randn(x1.shape[0])
zeros = np.zeros(y.shape)

gpns = GaussianProcessModel.NonStationaryGaussianProcessRegression(X, y)
gpns.computeGradient()
print(gpns.yGradient)

# Xq = np.random.rand(3, 2)
# yqGrad = interpn(X, gpns.yGradient, Xq)
# x1q = Xq[:, 0]
# x2q = Xq[:, 1]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection = '3d')
figg = plt.gcf()
figg.set_size_inches(19.2, 10.8)
ax.scatter(x1, x2, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig = plt.figure(2)
ax = plt.axes()
for i in range(y.shape[0]):
	ax.arrow(x1[i], x2[i], 0.01 * gpns.yGradient[i, 0], 0.01 * gpns.yGradient[i, 1], fc = 'k', ec = 'k')
plt.xlabel('x')
plt.ylabel('y')



plt.show()