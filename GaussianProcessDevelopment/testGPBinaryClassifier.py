import numpy as np
import GaussianProcessModel
import kernels
import responses
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Choose the squared exponential kernel for now
kernelchoice = kernels.m32iso
responsechoice = responses.logistic

nt = 200
nv = 100
k = 2

rangex = 5 # np.random.randint(2, 10)

# X  = (2*rangex*np.random.rand(n, k) - rangex) * np.array([2, 4])
# Xq = (2*rangex*np.random.rand(n, k) - rangex) * np.array([2, 4])

X  = (2*rangex*(np.random.rand(nt, k)) - rangex) * np.array([2, 4])
Xq = (2*rangex*(np.random.rand(nv, k)) - rangex) * np.array([2, 4])

x1 = X[:, 0]
x2 = X[:, 1]
xq1 = Xq[:, 0]
xq2 = Xq[:, 1]
r = np.sqrt(x1**2 + x2**2)
rmax = rangex/2
rOff1 = np.sqrt((x1 - rmax)**2 + (x2 - 3*rmax)**2)
rOff2 = np.sqrt((x1 + 2*rmax)**2 + (x2 + 2*rmax)**2)


y = np.ones(nt)

y[r < rmax] = -1
y[rOff1 < rmax] = -1
y[rOff2 < 2*rmax] = -1

# print('X:')
# print(X)
# print('\n\n')
# print('y')
# print(y)
# print('Kernel Parameters')
# print(kernelchoice.theta)


gpbc = GaussianProcessModel.GaussianProcessBinaryClassifier(X, y, kernel = kernelchoice, response = responsechoice)
gpbc.setInitialKernelHyperparameters(rangex/5 * np.ones(gpbc.getKernelHyperparameters().shape))
# gpbc.setInitialKernelHyperparameters(np.array([3, 1.5, 1.]))
gpbc.learn(train = True)
print('Finished Learning')
print(rangex)
piq = gpbc.predict(Xq)

yq = -np.ones(nv)
yq[piq > 0.5] = 1


circle1 = plt.Circle((0, 0), rmax, color = 'c', alpha = 0.2)
circle2 = plt.Circle((rmax, 3*rmax), rmax, color = 'c', alpha = 0.2)
circle3 = plt.Circle((-2*rmax, -2*rmax), 2*rmax, color = 'c', alpha = 0.2)

fig1 = plt.figure(figsize = (18, 18))
fig = plt.gcf()
fig.set_size_inches(19.2, 10.8)
plt.scatter(x1, x2, c = y, cmap = cm.gray)
plt.title('Training Labels')
plt.xlabel('x')
plt.ylabel('y')
fig = plt.gcf()
fig.gca().add_artist(circle1)
fig.gca().add_artist(circle2)
fig.gca().add_artist(circle3)
plt.colorbar()
plt.savefig('training.png', bbox_inches = 'tight')

# fig2 = plt.figure()
# plt.scatter(xq1, xq2, c = piq, cmap = cm.gray)
# plt.title('Probability of Validation Labels being 1')
# plt.xlabel('x')
# plt.ylabel('y')
# fig = plt.gcf()
# fig.gca().add_artist(circle1)
# plt.colorbar()

circle1 = plt.Circle((0, 0), rmax, color = 'c', alpha = 0.2)
circle2 = plt.Circle((rmax, 3*rmax), rmax, color = 'c', alpha = 0.2)
circle3 = plt.Circle((-2*rmax, -2*rmax), 2*rmax, color = 'c', alpha = 0.2)
fig3 = plt.figure(figsize = (18, 18))
fig = plt.gcf()
fig.set_size_inches(19.2, 10.8)
plt.scatter(xq1, xq2, c = yq, cmap = cm.gray)
plt.title('Predicted Validation Labels')
plt.xlabel('x')
plt.ylabel('y')
fig = plt.gcf()
fig.gca().add_artist(circle1)
fig.gca().add_artist(circle2)
fig.gca().add_artist(circle3)
plt.colorbar()
plt.savefig('validation.png', bbox_inches = 'tight')
# fig4 = plt.figure()
# plt.scatter(xq1, xq2, c = gpbc.f, cmap = cm.gray)
# plt.title('Predicted Validation Labels')
# plt.xlabel('x')
# plt.ylabel('y')
# fig = plt.gcf()
# fig.gca().add_artist(circle1)
# plt.colorbar()


plt.show()



