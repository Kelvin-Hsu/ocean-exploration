import numpy as np
import GaussianProcessModel
import matplotlib.pyplot as plt
import kernels
import responses

# def func(x):
# 	y = (x + 5)**2 * np.sin(0.5*(x + 5)**2)
# 	return(y)

def func(x):
	# x1 = x[x < 0]
	# x2 = x[x >= 0]
	# y1 = 10 * np.exp(x1) * np.sin(x1)

	# y = x**2 + 10*np.exp(-x**2) - 10*np.exp(-(x + 2)**2) + (x + 5)**2 * np.sin(0.8*(x + 5)**3)
	y = (x + 5)**2 * np.sin(0.8*(x + 5))**2
	y[x > -1] = y[x > -1] + 20
	y[x > 1] = y[x > 1] - 20
	return(y)

# def func(x):
# 	y = x**2
# 	return(y)

noiseLevel = 0.1
numberOfTrainingPoints = 80
halfRange = 5
x = 2 * halfRange * (1 - (np.random.rand(numberOfTrainingPoints))) - halfRange
x = np.append(x, np.array([-1.1, -0.9, 0.9, 1.1]))
y = func(x) + noiseLevel * np.random.randn(x.shape[0])


xq = np.linspace(x.min(), x.max(), num = 400)

X = x[:, np.newaxis]
Xq = xq[:, np.newaxis]

gpr = GaussianProcessModel.NonStationaryGaussianProcessRegression(X, y)

dx = x[1] - x[0]

yGradientEst = np.gradient(y, dx)


GaussianProcessModel.VERBOSE = True
gpr.learn(relearn = 1, optimiseLengthHyperparams = False)

lengthScales = gpr.predictLengthScales(X)
lengthScalesQ = gpr.predictLengthScales(Xq)

(yqExp, yqVar) = gpr.predict(Xq)
yqStd = np.sqrt(yqVar.diagonal())
yqL = yqExp - 2 * yqStd
yqU = yqExp + 2 * yqStd

(yqExpStat, yqVarStat) = gpr.gpStationary.predict(Xq)
yqStdStat = np.sqrt(yqVarStat.diagonal())
yqLStat = yqExpStat - 2 * yqStdStat
yqUStat = yqExpStat + 2 * yqStdStat

yqTrue = func(xq)

NUMPLTS = 5

plt.figure(1)
plt.subplot(NUMPLTS, 1, 1)
plt.scatter(x, y, c = 'g', label = 'Raw Data')
plt.plot(xq, yqTrue, c = 'g', label = 'Actual Function')
plt.plot(xq, yqExp, c = 'b', label = 'GP Predicted Function')
plt.fill_between(xq, yqU, yqL, facecolor = 'cyan', alpha = 0.2)
plt.title('Non-Stationary Gaussian Process Result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper left')

plt.subplot(NUMPLTS, 1, 2)
plt.scatter(x, y, c = 'g', label = 'Raw Data')
plt.plot(xq, yqTrue, c = 'g', label = 'Actual Function')
plt.plot(xq, yqExpStat, c = 'b', label = 'GP Predicted Function')
plt.fill_between(xq, yqUStat, yqLStat, facecolor = 'cyan', alpha = 0.2)
plt.title('Stationary Gaussian Process Result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper left')

plt.subplot(NUMPLTS, 1, 3)
plt.scatter(x, lengthScales, c = 'c')
plt.plot(xq, lengthScalesQ)
plt.plot(xq, gpr.stationaryLengthScale * np.ones(xq.shape), c = 'g')
plt.title('Underlying Length Scale')
plt.xlabel('x')
plt.ylabel('Length Scale')

plt.subplot(NUMPLTS, 1, 4)
plt.scatter(x, 1/lengthScales, c = 'c')
plt.plot(xq, 1/lengthScalesQ)
plt.plot(xq, 1/gpr.stationaryLengthScale * np.ones(xq.shape), c = 'g')
plt.title('Underlying Spatial Frequency')
plt.xlabel('x')
plt.ylabel('Length Scale')

plt.subplot(NUMPLTS, 1, 5)
plt.scatter(x, gpr.yGradient[:, 0], c = 'c')
plt.title('Instantaneous Gradient Prediction')
plt.xlabel('x')
plt.ylabel('dy/dx')

print('Sensitivity:', gpr.getSensitivity())
print('Stationary Length Scale:', gpr.getStationaryLengthScale())
print('Noise Level:', gpr.getNoiseLevel())
print('Length Scale Kernel Hyperparameters:', gpr.getLengthScaleKernelHyperparams())
print('Length Scale Noise Level:', gpr.getLengthScaleNoiseLevel())
print('Length Scale GP Length Scale Factor From Stationary GP Length Scale:', gpr.getLengthScaleKernelHyperparams()[1]/gpr.getStationaryLengthScale())
plt.show()
