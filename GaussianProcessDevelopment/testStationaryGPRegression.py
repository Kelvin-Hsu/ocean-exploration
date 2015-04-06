import numpy as np
import GaussianProcessModel
import matplotlib.pyplot as plt
import kernels

x = np.linspace(-5, 5, num = 20)
x = np.append(x, x)
y = x**2 * np.sin(x) + 2*np.random.randn(x.shape[0])
xq = np.linspace(-5, 5, num = 100)

X = x[:, np.newaxis]
Xq = xq[:, np.newaxis]

gpr = GaussianProcessModel.GaussianProcessRegression(X, y)

gpr.learn()

(yqExp, yqVar) = gpr.predict(Xq)
yqStd = np.sqrt(yqVar.diagonal())
yqL = yqExp - 2 * yqStd
yqU = yqExp + 2 * yqStd

plt.scatter(x, y, c = 'g')
plt.plot(xq, yqExp)
plt.fill_between(xq, yqU, yqL, facecolor = 'cyan', alpha = 0.2)
plt.show()