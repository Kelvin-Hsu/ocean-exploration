import numpy as np
import GaussianProcessModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import kernels
import responses


def mesh2data(Xmesh, Ymesh, Zmesh = None):

	(xLen, yLen) = Xmesh.shape
	n = xLen * yLen
	data = np.zeros((n, 2))
	data[:, 0] = np.reshape(Xmesh, n)
	data[:, 1] = np.reshape(Ymesh, n)

	if Zmesh == None:
		return(data)
	else:
		z = np.reshape(Zmesh, n)
		return(data, z)

def data2mesh(data, meshshape, zs = None):

	x = data[:, 0]
	y = data[:, 1]

	Xmesh = np.reshape(x, meshshape)
	Ymesh = np.reshape(y, meshshape)

	if zs == None:
		return(Xmesh, Ymesh)
	else:
		
		if type(zs) == int:
			Zmesh = np.reshape(zs, meshshape)
		else:
			Zmesh = zs.copy()
			for i in range(len(zs)):
				Zmesh[i] = np.reshape(zs[i], meshshape)
		return(Xmesh, Ymesh, Zmesh)

# def func(X):

# 	x1 = X[:, 0]
# 	x2 = X[:, 1]
# 	y = (x1 + 5)**2 * np.sin(0.8*(x1 + 5))**2 + 5*x2 + 1
# 	y[x1 > -1] = y[x1 > -1] + 20
# 	y[x1 > 1] = y[x1 > 1] - 20
# 	return(y)

def func(X):

	x1 = X[:, 0]
	x2 = X[:, 1]
	y = x1 + x2
	y[x1 + x2 > 0] = y[x1 + x2 > 0] + 10
	return(y)

noiseLevel = 1
numberOfTrainingPoints = 250
numberOfValidationPoints = 500

halfRange = 5
k = 2
X = 2 * halfRange * (1 - (np.random.rand(numberOfTrainingPoints, k))) - halfRange
x1 = X[:, 0]
x2 = X[:, 1]
y = func(X) # + # noiseLevel * np.random.randn(x.shape[0])

xq = np.linspace(-halfRange, halfRange, num = int(np.sqrt(numberOfValidationPoints)))

(xq1Mesh, xq2Mesh) = np.meshgrid(xq, xq)
meshshape = xq1Mesh.shape

Xq = mesh2data(xq1Mesh, xq2Mesh)
yqTrue = func(Xq)

gpr = GaussianProcessModel.NonStationaryGaussianProcessRegression(X, y)

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

NUMPLTS = 5

(_, _, listOfMeshs) = data2mesh(Xq, meshshape, [yqExp, yqStd, yqExpStat, yqStdStat, yqTrue, lengthScalesQ])

yqExpMesh = listOfMeshs[0]
yqStdMesh = listOfMeshs[1]
yqExpStatMesh = listOfMeshs[2]
yqStdStatMesh = listOfMeshs[3]
yqTrueMesh = listOfMeshs[4]
lengthScalesQMesh = listOfMeshs[5]

minYplt = np.min(np.array([yqTrueMesh.min(), yqExpMesh.min(), yqExpStatMesh.min()])) - 1

print(listOfMeshs)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(x1, x2, y, c = 'g', label = 'Raw Data')
ax1.plot_surface(xq1Mesh, xq2Mesh, yqTrueMesh, color = 'green', label = 'True Surface')
plt.title('Ground Truth')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.legend()

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection = '3d')
ax2.scatter(x1, x2, y, c = 'g', label = 'Raw Data')
ax2.plot_surface(xq1Mesh, xq2Mesh, yqExpMesh, cmap = cm.coolwarm, label = 'GP Predicted Surface')
ax2.plot_surface(xq1Mesh, xq2Mesh, yqTrueMesh, color = 'green', alpha = 0.1, label = 'True Surface')
levels = np.linspace(np.min(yqStdMesh),np.max(yqStdMesh), 50)
contourPlt = ax2.contour(xq1Mesh, xq2Mesh, yqStdMesh, levels, zdir = 'z', offset = minYplt, cmap = cm.coolwarm)
fig2.colorbar(contourPlt, shrink = 0.5, aspect = 5)
plt.title('Non-Stationary Gaussian Process Prediction')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.legend()

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection = '3d')
ax3.scatter(x1, x2, y, c = 'g', label = 'Raw Data')
ax3.plot_surface(xq1Mesh, xq2Mesh, yqExpStatMesh, cmap = cm.coolwarm, label = 'GP Predicted Surface')
ax3.plot_surface(xq1Mesh, xq2Mesh, yqTrueMesh, color = 'green', alpha = 0.1, label = 'True Surface')
levels = np.linspace(np.min(yqStdStatMesh),np.max(yqStdStatMesh), 50)
contourPlt = ax3.contour(xq1Mesh, xq2Mesh, yqStdStatMesh, levels, zdir = 'z', offset = minYplt, cmap = cm.coolwarm)
fig3.colorbar(contourPlt, shrink = 0.5, aspect = 5)
plt.title('Stationary Gaussian Process Prediction')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.legend()

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111, projection = '3d')
ax4.plot_surface(xq1Mesh, xq2Mesh, lengthScalesQMesh, cmap = cm.coolwarm)
ax4.plot_surface(xq1Mesh, xq2Mesh, gpr.getStationaryLengthScale() * np.ones(lengthScalesQMesh.shape), color = 'green', alpha = 0.1)
plt.title('Underlying Length Scale')
ax4.set_xlabel('x1')
ax4.set_ylabel('x2')
ax4.set_zlabel('y')

# fig3 = plt.figure(3)
# ax3 = fig2.add_subplot(111, projection = '3d')
# ax3.scatter(x1, x2, y, c = 'g', label = 'Raw Data')
# ax3.plot_surface(xq1Mesh, xq2Mesh, yqExpStatMesh, rstride = 1, cstride = 1, facecolors = yqStdStatMesh, linewidth = 0, antialiased = False, label = 'GP Predicted Surface')
# plt.title('Stationary Gaussian Process Prediction')
# ax3.set_xlabel('x1')
# ax3.set_ylabel('x2')
# ax3.set_zlabel('y')
# ax3.legend()

# plt.figure(1)
# plt.subplot(NUMPLTS, 1, 1)
# plt.scatter(x, y, c = 'g', label = 'Raw Data')
# yqTrue = func(xq)
# plt.plot(xq, yqTrue, c = 'g', label = 'Actual Function')
# plt.plot(xq, yqExp, c = 'b', label = 'GP Predicted Function')
# plt.fill_between(xq, yqU, yqL, facecolor = 'cyan', alpha = 0.2)
# plt.title('Non-Stationary Gaussian Process Result')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.subplot(NUMPLTS, 1, 2)
# plt.scatter(x, y, c = 'g', label = 'Raw Data')
# yqTrue = func(xq)
# plt.plot(xq, yqTrue, c = 'g', label = 'Actual Function')
# plt.plot(xq, yqExpStat, c = 'b', label = 'GP Predicted Function')
# plt.fill_between(xq, yqUStat, yqLStat, facecolor = 'cyan', alpha = 0.2)
# plt.title('Stationary Gaussian Process Result')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.subplot(NUMPLTS, 1, 3)
# plt.scatter(x, lengthScales, c = 'c')
# plt.plot(xq, lengthScalesQ)
# plt.plot(xq, gpr.stationaryLengthScale * np.ones(xq.shape), c = 'g')
# plt.title('Underlying Length Scale')
# plt.xlabel('x')
# plt.ylabel('Length Scale')

# plt.subplot(NUMPLTS, 1, 4)
# plt.scatter(x, 1/lengthScales, c = 'c')
# plt.plot(xq, 1/lengthScalesQ)
# plt.plot(xq, 1/gpr.stationaryLengthScale * np.ones(xq.shape), c = 'g')
# plt.title('Underlying Spatial Frequency')
# plt.xlabel('x')
# plt.ylabel('Length Scale')

# plt.subplot(NUMPLTS, 1, 5)
# plt.scatter(x, gpr.yGradient[:, 0], c = 'c')
# plt.title('Instantaneous Gradient Prediction')
# plt.xlabel('x')
# plt.ylabel('dy/dx')

# print('Sensitivity:', gpr.getSensitivity())
# print('Stationary Length Scale:', gpr.getStationaryLengthScale())
# print('Noise Level:', gpr.getNoiseLevel())
# print('Length Scale Kernel Hyperparameters:', gpr.getLengthScaleKernelHyperparams())
# print('Length Scale Noise Level:', gpr.getLengthScaleNoiseLevel())
# print('Length Scale GP Length Scale Factor From Stationary GP Length Scale:', gpr.getLengthScaleKernelHyperparams()[1]/gpr.getStationaryLengthScale())
plt.show()
