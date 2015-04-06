import numpy as np
import GaussianProcessModel
import matplotlib.pyplot as plt
import kernels

# x = np.linspace(-5, 5, num = 50)
# x = np.append(x, x)
# y = x**2 * np.sin(x) + 0.2*np.random.randn(x.shape[0])
y = np.array([0, 0.1, -0.1, 0, 0, 0.2, 0.3, 0.35, 0.4, 0.2, 0.1, -0.2, -4, -6, -7, -8, -10, -11, -10, -5, -5, -4.9, -4.8, -3, -1, 0, 6, 7, 8, 9, 10, 10, 10, 10, 10.1, 4, 2, 1, 0, 0, 0, 10, 20, 30, 60, 1, -10, -10.1, -10.2, -5.0, 0, 0])
x = np.linspace(-5, 5, num = y.shape[0])

print(y)

xq = np.linspace(-5, 5, num = 200)

X = x[:, np.newaxis]
Xq = xq[:, np.newaxis]

gpr = GaussianProcessModel.NonStationaryGaussianProcessRegression(X, y)

dx = x[1] - x[0]

yGradientEst = np.gradient(y, dx)

print(y.shape[0])
print(yGradientEst.shape[0])
print(gpr.yGradient.shape[0])
print(np.concatenate((gpr.yGradient, yGradientEst[:, np.newaxis]), axis = 1))
print(gpr.unscaledLengthScale)


gpr.learn()

lengthScales = gpr.getLengthScales()
lengthScalesQ = gpr.getLengthScales(Xq)

(yqExp, yqVar) = gpr.predict(Xq)
yqStd = np.sqrt(yqVar.diagonal())
yqL = yqExp - 2 * yqStd
yqU = yqExp + 2 * yqStd

plt.figure(1)
plt.scatter(x, y, c = 'g')
plt.plot(xq, yqExp)
plt.fill_between(xq, yqU, yqL, facecolor = 'cyan', alpha = 0.2)

plt.figure(2)
plt.scatter(x, lengthScales, c = 'c')
plt.plot(xq, lengthScalesQ)
plt.show()

