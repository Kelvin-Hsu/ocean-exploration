import numpy as np
import nlopt
import scipy.linalg as la
import kernels
import time

VERBOSE = False

# This finds the lower cholesky decomposition of a matrix and provides jittering if needed
def choleskyjitter(A):
	
    try:
        return(la.cholesky(A, lower = True))
    except Exception:
        pass

    n = len(A)
    maxscale = 10*np.sum(A.diagonal())
    minscale = min(1/64, maxscale/1024)
    scale = minscale

    if VERBOSE:
    	print('\t', 'Jittering...')

    while scale < maxscale:

        try:
        	jitA = scale * np.diag(np.random.rand(n))
        	L = la.cholesky(A + jitA, lower = True)
        	return(L)
        except Exception as e:
        	scale += minscale

    raise ValueError("Jittering failed")

# Here are some useful constants to be used in the Gaussian Process Class
INITIALISED = 0
LEARNED = 1
MISSING_KERNEL_MSG = "Please specify a kernel to use from kernels.py with the 'setKernelFunction()' method of your GP class."
HAS_NOT_LEARNED_MSG = "This Gaussian Process has not learned from the data yet. See the 'learn()' method of your GP class."
WRONG_PARAM_LENGTH_MSG = "Incorrect length of parameters specified."
POSITIVE_SIGMA_REQUIRED_MSG = "Sigma must be non-negative."

# This is the Gaussian Process Regression Class
class GaussianProcessRegression:

	# Initialise the Gaussian Process
	# Note that it is necessary to initialise the data
	# Previously, I had 'None' as the default value of 'kernel',
	# but I think it may make more sense to just default it to a commonly used kernel,
	# such as Squared Exponential Kernel or Matern 3/2 Kernel
	def __init__(self, X, y, kernel = kernels.se):

		# Set the data 
		self.setData(X, y)

		# Set the kernel function if it is provided
		self.setKernelFunction(kernel)

		# Initial hyperparameters and noise level to start off from during learning
		self.kernelHyperparamsInitial = None
		self.sigmaInitial = None

		# Actual hyperparameters and noise level to be used for prediction
		self.kernelHyperparams = None
		self.sigma = None

		# The matrix S is defined to be
		# 		S := K + sigma^2 * I
		# where K is the Data Kernel Matrix, sigma is the noise level, and I is the identity matrix
		# I have chosen to store this instead of K because it offers (slight) computational advantages
		# But I could easily have chosen to store K instead
		self.S = None
		self.state = INITIALISED

	# This sets the data of the Gaussian Process Model
	# This is called once automatically during the class initialisation
	# So, if you call it again, by implementation this means that you are overwriting the original data
	# By default, the class state will go back to being INITIALISED, even if it has LEARNED before
	# If the model has learned already and you would like to use the previous learning result (if you expect the previous model will work well with this new data), you can just set 'learned' to 'True'
	def setData(self, X, y):

		# Set the data matrix X and observed output data vector y
		# We do some basic whitening here by just making sure our output data has zero mean
		# We only add the mean back during the prediction stage
		# Otherwise, all analysis is done on the slightly whitened data vector y
		self.X = X.copy()
		self.yMean = np.mean(y)
		self.y = y.copy() - self.yMean

		# This is the number of observed data points and number of features from the data matrix
		(self.n, self.k) = np.shape(X)

		# We can initialise the appropriate identity matrix here for later use
		self.I = np.eye(self.n)

		# If the user wants to use the previous learning result, let them do it
		# Otherwise, reset the class state to INITIALISED
		
	def setResponseData(self, y):

		# This will assume that the length of 'y' is the same as the number of data points we have
		self.yMean = np.mean(y)
		self.y = y.copy() - self.yMean

	# This adds new data on top of the original data into the model 
	def addData(self, X, y):

		# If the Gaussian Process Model has already been trained, we will use our fast cholesky update method to update the Cholesky Decomposition Matrix
		if self.state == LEARNED:

			# This is the number of observations of the incoming data
			nNew = X.shape[0]

			# First, we would have to update our S matrix
			# Again, the matrix S is defined to be
			# 		S := K + sigma^2 * I
			# where K is the Data Kernel Matrix, sigma is the noise level, and I is the identity matrix
			S = self.S.copy()
			Sn = self._kernel(self.X, X)
			Snn = self._kernel(X, X) + self.sigma**2 * np.eye(nNew)
			top = np.concatenate((S, Sn), axis = 1)
			bottom = np.concatenate((Sn.T, Snn), axis = 1)
			self.S = np.concatenate((top, bottom), axis = 0)

			# Now, we can use our fast cholesky update algorithm to find our lower cholesky decomposition without recomputing everything
			L = self.L.copy()
			Ln = la.solve_triangular(L, Sn, lower = True).T
			On = np.zeros(Ln.shape).T
			Lnn = choleskyjitter(Snn - Ln.dot(Ln.T))
			top = np.concatenate((L, On), axis = 1)
			bottom = np.concatenate((Ln, Lnn), axis = 1)
			self.L = np.concatenate((top, bottom), axis = 0)

		# In any case, we will just add the data into the model
		self.X = np.concatenate((self.X, X))

		# For the y data vector, we need to again make sure we slightly whiten the data to zero mean
		self.y += self.yMean
		self.y = np.concatenate((self.y, y))
		self.yMean = np.mean(self.y)
		self.y -= self.yMean

		# This is the number of observed data points and number of features from the data matrix
		(self.n, self.k) = np.shape(self.X)

		# We can initialise the appropriate identity matrix here for later use
		self.I = np.eye(self.n)

	# This obtains the data currently used by the model
	def getData(self):

		return(self.X.copy(), self.y + self.yMean)

	# This function is only used by 'self.addData(X, y)' to retain the trained properties of the model from the last training session
	# This function can also be externally whenever the user wants to change the state of the model to 'learned' or 'trained'
	def usePreviousLearningResult(self):

		# We can only force the model state to be LEARNED it has actually been trained before
		if self.kernelHyperparams != None & self.sigma != None:

			# If so, we would have to recompute the cholesky decomposition again
			self._prepareCholesky()
			self.state = LEARNED

		# Otherwise, this function will raise an error
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This function sets the kernel function that this gaussian process model would be using
	def setKernelFunction(self, kernelfunction):

		# This is a reference to the kernel function we will be using, which is used to call the function
		self._kernel = kernelfunction

		# We need to remember to set the correct number of feature dimensions for the kernel
		kernels.setNumberOfParams(self._kernel, self.k)

		# For convenience, we will store our own copy of the names of the kernel hyperparameters
		self.kernelHyperparamNames = self._kernel.thetaNames

	# This function can be called before 'self.learn()' to set starting point of the hyperparameters of the kernel before training
	def setInitialKernelHyperparams(self, hyperparams):

		# Just to be safe, we will make sure the number of hyperparameters have been set correctly
		kernels.setNumberOfParams(self._kernel, self.k)
		
		# If the number of initial parameters supplied does not match the numbers needed, raise an error
		if len(hyperparams) != len(self._kernel.thetaInitial):
			raise ValueError(WRONG_PARAM_LENGTH_MSG)
		
		# Otherwise, simply set the initial parameters
		self.kernelHyperparamsInitial = hyperparams.copy()

	# This function can be called before 'self.learn()' to set starting point of the noise level before training
	def setInitialSigma(self, sigma):

		# This needs to be a positive number, obviously
		if sigma < 0:
			raise ValueError(POSITIVE_SIGMA_REQUIRED_MSG)

		# Simply set the starting noise level
		self.sigmaInitial = sigma

	# This function returns the current or lastly used starting point of the initial kernel hyperparameters
	def getInitialKernelParams(self):

		return(self.kernelHyperparamsInitial.copy())

	# This function returns the current or lastly used starting point of the initial noise level
	def getInitialSigma(self):

		return(self.sigmaInitial)

	# This function returns the name of the kernel used
	def getKernelName(self):

		# Return the name of the kernel
		return(self._kernel.name)

	# This function returns the name of the kernel hyperparameters
	def getKernelHyperparamNames(self):

		# Return the name of the kernel hyperparameters
		return(self.kernelHyperparamNames.copy())

	# This function can be called to directly set a particular training or learning result (hyperparameters and noise level) to the model
	# A common usage of this function is when the user already knows the appropriate hyperparameters and noise levels for experience, and want to skip the potentially lengthy training time
	def setLearningResult(self, hyperparams, sigma):

		# Just to be safe, we will make sure the number of hyperparameters have been set correctly
		kernels.setNumberOfParams(self._kernel, self.k)

		# Make sure the number of hyperparameters proposed is correct
		if len(hyperparams) != len(self._kernel.theta):
			raise ValueError(WRONG_PARAM_LENGTH_MSG)

		# Make sure the value of noise level proposed is valid
		if sigma < 0:
			raise ValueError(POSITIVE_SIGMA_REQUIRED_MSG)

		# Set the learning results (hyperparameters)
		self.kernelHyperparams = hyperparams.copy()
		self._kernel.theta = hyperparams.copy()

		# Set the learning results (noise level)
		self.sigma = sigma

		# Compute the lower cholesky decomposition for prediction purposes
		self._prepareCholesky()

		# We have finished 'learning'
		self.state = LEARNED
		
	# This function is the training part of the gaussian process model
	# The keyword 'sigma' can be set to a particular value so that the noise level is kept the same during learning and the hyperparameters will be trained to this particular noise level
	# The keyword 'sigmaMax' can be set to a particular value to make sure the gaussian process will never train to a noise level above this threshold
	def learn(self, sigma = None, sigmaMax = None, thetaLowerBound = None):

		# VERBOSE
		if VERBOSE:
			np.set_printoptions(precision = 2)

		# If the model has already been trained before, we can start off our learning from these values
		if self.state == LEARNED:
			self.kernelHyperparamsInitial = self.kernelHyperparams.copy()
			self.sigmaInitial = self.sigma

		# If there is no initial starting hyperparameters provided, we will just use the default one from the kernels
		if self.kernelHyperparamsInitial == None:
			kernels.setNumberOfParams(self._kernel, self.k)
			self.kernelHyperparamsInitial = self._kernel.theta.copy()

		# If the user did not specify to fix the noise level, then do the following
		if sigma == None:

			# The number of parameters in the optimisation/learning stage is 1 more (with the noise level) then the number of hyperparameters in the kernel
			kparam = self._kernel.theta.shape[0] + 1

			# If the user did not specify a initial noise level to start from, default it to some arbitary value
			if self.sigmaInitial == None:
				sigmaInitial = np.std(self.y) * 1e-3

			# Otherwise, use the user-provided value
			else:
				sigmaInitial = self.sigmaInitial

		# Otherwise, if the user did specify to fix the noise level, then do the following
		else:

			# Make sure the value of noise level proposed is valid
			if sigma < 0:
				raise ValueError(POSITIVE_SIGMA_REQUIRED_MSG)

			# The number of parameters in the optimisation/learning stage is the number of hyperparameters in the kernel
			kparam = self._kernel.theta.shape[0]

			# Fix the noise level
			self.sigma = sigma

		# This function calculates the negative log(evidence) without the constant term
		def negLogEvidence(theta, grad):

			self._kernel.theta = theta[0:-1].copy()
			self.sigma = theta[-1]

			self._prepareCholesky()
			alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
			negLogEvidenceValue = 0.5 * self.y.dot(alpha) + np.sum(np.log(self.L.diagonal()))
			
			# VERBOSE
			if VERBOSE:
				if negLogEvidence.lastPrintedSec != time.gmtime().tm_sec:
					negLogEvidence.lastPrintedSec = time.gmtime().tm_sec
					print('\t', theta, negLogEvidenceValue)

			return(negLogEvidenceValue)

		# VERBOSE
		if VERBOSE:
			negLogEvidence.lastPrintedSec = -1

		# This function calculates the negative log(evidence) without the constant term while keeping sigma fixed
		def negLogEvidenceNoSigma(theta, grad):

			self._kernel.theta = theta.copy()

			self._prepareCholesky()
			alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
			negLogEvidenceValue = 0.5 * self.y.dot(alpha) + np.sum(np.log(self.L.diagonal()))
			
			# VERBOSE
			if VERBOSE:
				if negLogEvidenceNoSigma.lastPrintedSec != time.gmtime().tm_sec:
					negLogEvidenceNoSigma.lastPrintedSec = time.gmtime().tm_sec
					print('\t', theta, negLogEvidenceValue)

			return(negLogEvidenceValue)

		# VERBOSE
		if VERBOSE:
			negLogEvidenceNoSigma.lastPrintedSec = -1

		# This is the constant involved in the computation of the log(evidence)
		evidenceConst = (self.n * np.log(2 * np.pi) / 2)



		# Initialise our optimiser with the right amount of parameters
		opt = nlopt.opt(nlopt.LN_BOBYQA, kparam)

		# Set a non-zero but small lower bound for computational safety
		if thetaLowerBound == None:
			opt.set_lower_bounds(1e-5 * np.ones(kparam))
		else:
			opt.set_lower_bounds(thetaLowerBound)

		# Set an appropriate tolerance level
		opt.set_ftol_rel(1e-3)
		opt.set_xtol_rel(1e-3)

		# If the user did not fix the noise level, do the following
		if sigma == None:

			# Set the upper bound for sigma if the user specified to
			if sigmaMax != None:
				ub = np.inf*np.ones(kparam)
				ub[-1] = sigmaMax
				opt.set_upper_bounds(ub)
			
			# Set the objective function
			opt.set_min_objective(negLogEvidence)

			# Set the initial parameters to be optimised
			thetaInitial = np.append(self.kernelHyperparamsInitial, sigmaInitial) + 1e-7

			# VERBOSE
			if VERBOSE:
				print('Params:\t', self.getKernelHyperparamNames())
				print('Format:\t [(--kernel-parameters--) (sigma)] negative(log(evidence) + ' + str(round(evidenceConst, 3)) + ')')

			# Obtain the final optimal parameters
			thetaFinal = opt.optimize(thetaInitial)

			# Obtain the optimal hyperparameters
			self._kernel.theta = thetaFinal[0:-1].copy()
			self.kernelHyperparams = self._kernel.theta.copy()

			# Obtain the optimal noise level
			self.sigma = thetaFinal[-1]

			# Store the final log(evidence) value
			self.logevidence = -negLogEvidence(thetaFinal, thetaFinal) - evidenceConst

		else:

			# Set the objective function
			opt.set_min_objective(negLogEvidenceNoSigma)

			# VERBOSE
			if VERBOSE:
				print('Sigma has been set to', sigma)
				print('Params:\t', self.getKernelHyperparamNames())
				print('Format:\t [(--kernel-parameters--)] negative(log(evidence) + ' + str(round(evidenceConst, 3)) + ')')

			# Obtain the final optimal hyperparameters
			self.kernelHyperparams = opt.optimize(self.kernelHyperparamsInitial)
			self._kernel.theta = self.kernelHyperparams.copy()

			# Store the final log(evidence) value
			self.logevidence = -negLogEvidenceNoSigma(self.kernelHyperparams, self.kernelHyperparams) - evidenceConst

		# Compute and prepare the lower cholesky decomposition matrix
		self._prepareCholesky()

		# We have finished learning
		self.state = LEARNED

		# VERBOSE
		if VERBOSE:
			print('Learned:\t Kernel HyperParameters:', self.kernelHyperparams, '| Sigma:', self.sigma, '| log(evidence):', self.logevidence)

	# This function is the prediction part of the gaussian process model
	# Use this on a query data matrix to find either the predicted expected latent function and its covariance or the expected output function and its covariance
	# By default, the covariance corresponds to the latent function
	# To set it to be corresponding to the output function, set 'predictObservations' to 'True'
	def predict(self, Xq, predictUncertainty = True, predictObservations = False):

		# If the model has not learned, we should not predict anything
		self.requireLearned()

		# Set the learned result
		self._kernel.theta = self.kernelHyperparams.copy()

		# Compute the prediction using the standard Gaussian Process Prediction Algorithm
		alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
		Kq = self._kernel(self.X, Xq)
		fqExp = np.dot(Kq.T, alpha) + self.yMean

		# Compute the covariance matrix using the standard Gaussian Process Prediction Algorithm if requested
		if predictUncertainty:

			v = la.solve_triangular(self.L, Kq, lower = True, check_finite = False)
			Kqq = self._kernel(Xq, Xq)
			fqVar = Kqq - np.dot(v.T, v)

		# Depending on if the user wants the one for latent function or output function, return the correct expected values and covariance matrix
		if predictUncertainty:
			if predictObservations:
				return(fqExp, fqVar + self.sigma**2 * np.eye(fqVar.shape[0]))
			else:
				return(fqExp, fqVar)
		else:
			return(fqExp)

	# This returns the hyperparameters of the kernel if the model has been trained
	def getKernelHyperparams(self):

		if self.state == LEARNED:
			return(self.kernelHyperparams.copy())
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This returns the noise level if the model has been trained
	def getSigma(self):

		if self.state == LEARNED:
			return(self.sigma)
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This returns the log(evidence) if the model has been trained
	def getLogEvidence(self):

		if self.state == LEARNED:
			try:
				return(self.logevidence)
			except AttributeError:
				return(None)
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This returns the raw state of the model in raw values
	# INITIALISED = 0
	# LEARNED = 1
	def getState(self):

		return(self.state)

	# This returns the state of the model in binary truth values
	def hasLearned(self):

		if self.state == LEARNED:
			return(True)
		else:
			return(False)

	# For debugging purposes, you can use this to stop the program at places where you require the gaussian process model to have been trained before proceeding
	def requireLearned(self):

		if self.state != LEARNED:
			raise ValueError(HAS_NOT_LEARNED_MSG)
			
	## --Below are private methods--

	# This prepares the lower cholesky decomposition of S
	# Again, the matrix S is defined to be
	# 		S := K + sigma^2 * I
	# where K is the Data Kernel Matrix, sigma is the noise level, and I is the identity matrix
	def _prepareCholesky(self):

		self.S = self._kernel(self.X, self.X) + self.sigma**2 * self.I
		self.L = choleskyjitter(self.S)




























































#### Research Code Below...
from scipy import stats
import responses

# This is the Gaussian Process Binary Classifier Class
class GaussianProcessBinaryClassifier:

	# Initialise the Gaussian Process Binary Classifier
	def __init__(self, X, y, kernel = kernels.m32, response = responses.logistic):

		# Set the training data of the binary classifier
		self.setData(X, y)

		# Set the kernel function of the binary classifier
		self.setKernelFunction(kernel)

		# Set the response (soft-max/sigmoid) function of the binary classifier
		self.setResponseFunction(response)

	# This sets the data we are training/learning on for the Gaussian Process Binary Classifier
	def setData(self, X, y):

		# Set the data matrix X and labels y
		self.X = X.copy()
		self.y = y.copy()

		# This is the number of observed data points and number of features from the data matrix
		(self.n, self.k) = np.shape(X)

		# We can initialise the appropriate identity matrix here for later use
		self.I = np.eye(self.n)

		# When we set a new kernel, any previous learning result should not really make much sense, so we would reset the model state to INITIALISED
		# However, if the user really wants to use a previous learning result, they are welcome to use 'self.usePreviousLearningResult()'
		self.state = INITIALISED

	# This function sets the kernel function that this gaussian process model would be using
	def setKernelFunction(self, kernelfunction):

		# This is a reference to the kernel function we will be using, which is used to call the function
		self._kernel = kernelfunction

		# We need to remember to set the correct number of feature dimensions for the kernel
		kernels.setNumberOfParams(self._kernel, self.k)

		# For convenience, we will store our own copy of the names of the kernel hyperparameters
		self.kernelHyperparamNames = self._kernel.thetaNames

		# We will calculate the kernel matrix here
		self.K = self._kernel(self.X, self.X)

		# Use the kernel recommended initial hyperparameters as the initial hyperparameters
		self.kernelHyperparams = self._kernel.theta.copy()
		self.kernelHyperparamsInitial = self._kernel.thetaInitial.copy()

		# When we set a new kernel, any previous learning result should not really make much sense, so we would reset the model state to INITIALISED
		self.state = INITIALISED

	def setResponseFunction(self, responsefunction):

		# Set the response (likelihood) function of our binary classifier
		self._response = responsefunction

		# When we set a new response function, any previous learning result should not really make much sense, so we would reset the model state to INITIALISED
		self.state = INITIALISED

	def setInitialKernelHyperparameters(self, thetaInitial):

		if thetaInitial.shape[0] == self._kernel.theta.shape[0]:
			self.kernelHyperparamsInitial = thetaInitial.copy()
		else:
			raise ValueError(WRONG_PARAM_LENGTH_MSG)

	def setKernelHyperparameters(self, theta):

		if theta.shape[0] == self._kernel.theta.shape[0]:
			self.kernelHyperparams = theta.copy()
			self._kernel.theta = theta.copy()
		else:
			raise ValueError(WRONG_PARAM_LENGTH_MSG)

	def getInitialKernelHyperparameters(self):

		return(self.kernelHyperparamsInitial)

	def getKernelHyperparameters(self):

		return(self.kernelHyperparams)

	def learn(self, train = True, ftol = 1e-10):

		# This is the closure to find the log-marginal-likelihood
		def logEvidence(theta, grad):

			# We will calculate the kernel matrix here
			self._kernel.theta = theta.copy()
			self.K = self._kernel(self.X, self.X)

			# Initialise the latent function in a previous trial as anything that is far away from the initial latent function with respect to the tolerance level
			fprev = logEvidence.f + 1e2*ftol
			
			# Use Newton–Raphson Method to learn the latent function
			while la.norm(logEvidence.f - fprev) > ftol:

				# Note the current latent function as the previous latent function
				fprev = logEvidence.f.copy()

				# Compute the derivative of the log-likelihood
				logEvidence.dloglikelihood = self._response.gradientLogLikelihood(self.y, logEvidence.f)

				# Compute the negative second derivative of the log-liklihood and its square root
				# Those are diagonal matrices
				logEvidence.W = -self._response.hessianLogLikelihood(self.y, logEvidence.f)
				logEvidence.Wsqrt = np.sqrt(logEvidence.W)

				# Re-learn the latent function
				B = self.I + logEvidence.Wsqrt.dot(self.K).dot(logEvidence.Wsqrt)
				logEvidence.L = choleskyjitter(B)
				b = logEvidence.W.dot(logEvidence.f) + logEvidence.dloglikelihood
				c = la.solve_triangular(logEvidence.L, logEvidence.Wsqrt.dot(self.K).dot(b), lower = True, check_finite = False)
				a = b - logEvidence.Wsqrt.dot(la.solve_triangular(logEvidence.L.T, c, check_finite = False))
				logEvidence.f = self.K.dot(a)

			# Compute the log-marginal-liklihood or log-evidence of the learning result
			logEvidenceValue = -0.5 * a.dot(logEvidence.f) + self._response.logLikelihood(self.y, logEvidence.f) - np.sum(np.log(logEvidence.L.diagonal()))
			# logEvidenceValue2 = -0.5 * logEvidence.f.dot(la.solve(self.K, logEvidence.f, check_finite = False)) + np.log(self._response.likelihood(self.y, logEvidence.f)) - 0.5 * np.linalg.slogdet(B)[1]
			if VERBOSE:
				print(logEvidenceValue, theta)
			return(logEvidenceValue)

		# Initialise the latent function as a zero function
		logEvidence.f = np.zeros(self.n)

		# Initialise important quantities we can store for prediction
		logEvidence.W = self.I.copy()
		logEvidence.Wsqrt = self.I.copy()
		logEvidence.L = self.I.copy()
		logEvidence.dloglikelihood = np.zeros(self.n)

		# If we want to train the model to learn well, use the optimiser to optimise the log-marginal-likelihood
		if train:

			# Initialise our optimiser with the right amount of parameters
			kparams = self._kernel.theta.shape[0]
			opt = nlopt.opt(nlopt.LN_COBYLA, kparams)

			# Set a non-zero but small lower bound for computational safety
			opt.set_lower_bounds(np.append(1e-8 * np.ones(1), 1e-8 * np.ones(kparams - 1)))

			# Set an appropriate tolerance level
			opt.set_ftol_rel(1e-7)
			opt.set_xtol_rel(1e-7)

			# We want to maximise the log-marginal-likelihood or log-evidence
			opt.set_max_objective(logEvidence)

			# Learn the hyper-parameters
			thetaFinal = opt.optimize(self.kernelHyperparamsInitial)

			# Obtain the log-marginal-likelihood
			self.logMaginalLikelihood = opt.last_optimum_value()

			# Set the hyper-parameters
			self._kernel.theta = thetaFinal.copy()
			self.kernelHyperparams =  thetaFinal.copy()

		# Otherwise, if we just want to learn the latent function, do so
		else:

			# Obtain the log-marginal-likelihood without training for hyperparameters
			self.logMaginalLikelihood = logEvidence(self.kernelHyperparams, np.zeros(self.kernelHyperparams.shape))

		# Store the important quantities below for faster prediction
		self.f = logEvidence.f
		self.W = logEvidence.W
		self.Wsqrt = logEvidence.Wsqrt
		self.L = logEvidence.L
		self.dloglikelihood = logEvidence.dloglikelihood

		# We have finished learning
		self.state = LEARNED

	def predict(self, Xq):

		# We require the model already have learned before we perform prediction
		self.forceLearn()

		# Make sure we are using the right kernel hyperparameters
		self._alignParameters()

		# Perform the prediction to obtain the expected class probability
		Kq = self._kernel(self.X, Xq)
		fqExp = Kq.T.dot(self.dloglikelihood)
		v = la.solve_triangular(self.L, self.Wsqrt.dot(Kq), lower = True, check_finite = False)
		Kqq = self._kernel(Xq, Xq)
		fqVar = Kqq - np.dot(v.T, v)
		z = fqExp / np.sqrt(1 + fqVar.diagonal())
		piq = stats.norm.cdf(z)
		return(piq)

	# If a part of the program needs the model to have learned before we continue, this function can be called to stop execution if learning has not been done
	def requireLearned(self):

		# Stop the program if the model has not learned yet
		if self.state != LEARNED:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# If a part of the program needs the model to have learned before we continue, this function can be called to force the model to learn if it has not learned
	def forceLearn(self):

		# Stop the program if the model has not learned yet
		if self.state != LEARNED:

			# We don't need to optimise the learning here
			self.learn(train = False)

	def _alignParameters(self):

		self._kernel.theta = self.kernelHyperparams.copy()


## MORE RESEARCH CODE
from scipy.spatial.distance import cdist

def clipQuantiles(array, qs, qf):

	arraySorted = np.sort(array)
	iMin = int(0.1 * arraySorted.shape[0]) + 1
	iMax = int(0.9 * arraySorted.shape[0])
	arrayMin = arraySorted[iMin]
	arrayMax = arraySorted[iMax]
	array = np.clip(array, arrayMin, arrayMax)
	return(array)

## class 'NonStationaryGaussianProcessRegression'
# 
#  --Description--
#  This class implements the non-stationary version of the Gaussian Process Regression.
#  The model specifically uses a non-stationary squared exponential kernel
class NonStationaryGaussianProcessRegression:

	def __init__(self, X, y):

		# Set the data 
		self.setData(X, y)

		# Model the data with a stationary GP first
		self.learnStationaryModel()

		self.setInitialHyperLengthScale(15, relative = True)
		self.setMinHyperLengthScale(0.1, relative = True)

		# Initialise the maximum length scale
		self.setMaxLengthScale(np.Inf)

		# Initialise the minimum length scale
		self.setMinLengthScale(self.stationaryLengthScale/50)

		# Prepare a stationary Gaussian Process Regression Model for kriging the underlying length scale function
		self.updateLengthScaleModel()

	# This sets the data of the Gaussian Process Model
	# This is called once automatically during the class initialisation
	# So, if you call it again, by implementation this means that you are overwriting the original data
	def setData(self, X, y):

		# Set the data matrix X and observed output data vector y
		# We do some basic whitening here by just making sure our output data has zero mean
		# We only add the mean back during the prediction stage
		# Otherwise, all analysis is done on the slightly whitened data vector y
		self.X = X.copy()
		self.yMean = y.mean()
		self.y = (y.copy() - self.yMean)

		# This is the number of observed data points and number of features from the data matrix
		(self.n, self.k) = np.shape(X)

		# We can initialise the appropriate identity matrix here for later use
		self.I = np.eye(self.n)

	def setInitialHyperLengthScale(self, hyperLengthScaleInitial, relative = False):

		if relative:
			self.hyperLengthScaleInitial = self.gpStationary.getKernelHyperparams()[1] * hyperLengthScaleInitial
		else:
			self.hyperLengthScaleInitial =  hyperLengthScaleInitial

		self.gpLengthScale.setLearningResult(np.array([self.gpLengthScale.getKernelHyperparams()[0], self.hyperLengthScaleInitial]), 2)

	def setMinHyperLengthScale(self, hyperLengthScaleMin, relative = False):

		if relative:
			self.hyperLengthScaleMin = self.gpStationary.getKernelHyperparams()[1] * hyperLengthScaleMin
		else:
			self.hyperLengthScaleMin =  hyperLengthScaleMin

	def setSensitivity(self, alpha):

		# Set the sensitivity
		self.alpha = alpha

	def setNoiseLevel(self, sigma):

		# Set the noise level
		self.sigma = sigma

	def setMaxLengthScale(self, maxLengthScale):

		# Set the maximum length scale bound
		self.maxLengthScale = maxLengthScale

	def setMinLengthScale(self, minLengthScale):

		# Set the minimum length scale bound
		self.minLengthScale = minLengthScale

	def setStationaryLengthScale(self, stationaryLengthScale):

		# Set the stationary length scale
		self.stationaryLengthScale = stationaryLengthScale

	def getSensitivity(self):

		# Return the sensitivity
		return(self.alpha)

	def getNoiseLevel(self):

		# Return the noise level
		return(self.sigma)

	def getMaxLengthScale(self):

		# Return the maximum length scale bound
		return(self.maxLengthScale)

	def getMinLengthScale(self):

		# Return the minimum length scale bound
		return(self.minLengthScale)

	def getStationaryLengthScale(self):

		# Return the stationary length scale
		return(self.stationaryLengthScale)

	def getLengthScaleKernelHyperparams(self):

		return(self.gpLengthScale.kernelHyperparams)

	def getLengthScaleNoiseLevel(self):

		return(self.gpLengthScale.sigma)

	def getInitialHyperLengthScale(self, relative = False):

		if relative:
			return(self.hyperLengthScaleInitial/self.getStationaryLengthScale())
		else:
			return(self.hyperLengthScaleInitial)

	def getInitialHyperLengthScale(self, relative = False):

		if relative:
			return(self.hyperLengthScaleInitial/self.getStationaryLengthScale())
		else:
			return(self.hyperLengthScaleInitial)

	def getMinHyperLengthScale(self, relative = False):

		if relative:
			return(self.hyperLengthScaleMin/self.getStationaryLengthScale())
		else:
			return(self.hyperLengthScaleMin)

	def getCurrentHyperLengthScale(self, relative = False):

		if relative:
			return(self.getLengthScaleKernelHyperparams()[1]/self.getStationaryLengthScale())
		else:
			return(self.getLengthScaleKernelHyperparams()[1])
			
	def learnStationaryModel(self):

		# Generate a stationary gaussian process regression model on the data
		self.gpStationary = GaussianProcessRegression(self.X, self.y + self.yMean, kernels.seiso)

		# Let the process learn
		# self.gpStationary.setInitialKernelParams(np.array([1, self.maxLengthScale/2]))
		self.gpStationary.learn()

		# Initialise the non-stationary GP model with the results of the stationary GP model
		stationaryHyperparams = self.gpStationary.getKernelHyperparams()
		self.setSensitivity(stationaryHyperparams[0])
		self.setStationaryLengthScale(stationaryHyperparams[1])
		self.setNoiseLevel(self.gpStationary.getSigma())
		self.L = self.gpStationary.L.copy()

		# Initialise the non-stationary GP model with the length scales of the stationary GP model
		self.lengthScale = self.stationaryLengthScale * np.ones(self.y.shape)

		# Initialise a GP on the length scale
		self.gpLengthScale = GaussianProcessRegression(self.X, self.lengthScale, kernel = kernels.seiso)
		self.gpLengthScale.setLearningResult(self.gpStationary.getKernelHyperparams() * np.array([1, 50]), 2)

	# Given model with length scale, compute gradient
	def updateGradient(self):

		# Initialise the gradient matrix (gradient vector for each observation)
		self.yGradient = np.zeros(self.X.shape)

		lengthSteps = self.stationaryLengthScale * np.ones(self.y.shape)

		# This is the finite difference coefficients for first derivative approximation with accuracy level 2
		cuu = -1/12
		cu = 2/3
		cl = -cu
		cll = -cuu

		# Go through each feature dimension
		for i in range(self.k):

			# Initialise a step vector (which should contain zeros everywhere except the spatial step size at the dimension we are looking at)
			oneLengthStep = np.zeros(self.X.shape)
			oneLengthStep[:, i] = lengthSteps

			# Obtain the query points for gradient computation
			Xuu = self.X + 2*oneLengthStep
			Xu  = self.X +   oneLengthStep
			Xl  = self.X -   oneLengthStep
			Xll = self.X - 2*oneLengthStep

			# Estimate the latent function at these query locations
			yuu = self.predict(Xuu, predictUncertainty = False)
			yu = self.predict(Xu, predictUncertainty = False)
			yl = self.predict(Xl, predictUncertainty = False)
			yll = self.predict(Xll, predictUncertainty = False)

			# Estimate the gradient vector at these locations
			gradient = (cuu*yuu + cu*yu + cl*yl + cll*yll)/lengthSteps

			self.yGradient[:, i] = clipQuantiles(gradient, 0.1, 0.9)

	def updateLengthScaleModel(self):

		# Update the length scale GP model and train the model
		self.gpLengthScale.setResponseData(self.lengthScale)

		# Update the new length scale estimates
		self.lengthScale = self.predictLengthScales(self.X)

	def predictLengthScales(self, Xq):

		lengthScaleQuery = self.gpLengthScale.predict(Xq, predictUncertainty = False)

		return(np.clip(lengthScaleQuery, self.minLengthScale, self.maxLengthScale))

	def kernel(self, X1, X2):

		# Find the length scales at X1 and X2
		lengthScalesSq1 = self.predictLengthScales(X1)**2
		lengthScalesSq2 = self.predictLengthScales(X2)**2

		# Find the determinant of the length scale matrix at X1 and X2
		(detLengthScalesSq1, detLengthScalesSq2) = np.meshgrid(lengthScalesSq1, lengthScalesSq2)
		detLengthScalesSq1 = (detLengthScalesSq1.T) ** self.k
		detLengthScalesSq2 = (detLengthScalesSq2.T) ** self.k

		# Find the average length scales between X1 and X2
		averageLengthScalesSq = (lengthScalesSq1[:, np.newaxis] + lengthScalesSq2[:, np.newaxis].T)/2

		# Find the determinant of the average length scale matrix between X1 and X2
		detAverageLengthScalesSq = averageLengthScalesSq ** self.k

		# This is the exponent of the squared exponential kernel
		a = 0.5 * cdist(X1, X2, 'sqeuclidean')/averageLengthScalesSq

		# This is the sensitivity of the squared exponential kernel
		multiplier = self.alpha**2 * np.sqrt(np.sqrt(detLengthScalesSq1 * detLengthScalesSq2) / detAverageLengthScalesSq)

		# This is the non-stationary squared exponential kernel matrix
		return(multiplier * np.exp(-a))

	def learn(self, theta = None, maxtime = 30, relearn = 0, optimiseLengthHyperparams = False, _optimiseLengthHyperparams = False, _optimiseLengthSigma = False):

		if relearn > 0:
			optimiseLengthHyperparams = False
			_optimiseLengthHyperparams = False
			_optimiseLengthSigma = False

		# This function calculates the negative log(evidence) without the constant term
		def negLogEvidence(theta, grad):

			if _optimiseLengthHyperparams:
				self.lengthScale = theta[:-2]
				self.gpLengthScale.setLearningResult(theta[-2:], 0)
			else:
				self.lengthScale = theta.copy()

			self.updateLengthScaleModel()

			self.S = self.kernel(self.X, self.X) + self.sigma**2 * self.I
			self.L = choleskyjitter(self.S)

			alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
			negLogEvidenceValue = 0.5 * self.y.dot(alpha) + np.sum(np.log(self.L.diagonal()))
			
			# VERBOSE
			if VERBOSE:
				negLogEvidence.counter += 1

				if negLogEvidence.lastPrintedSec != time.gmtime().tm_sec:
					negLogEvidence.lastPrintedSec = time.gmtime().tm_sec
					if _optimiseLengthHyperparams:
						print(negLogEvidence.counter, '\t', self.lengthScale, self.stationaryLengthScale, self.gpLengthScale.kernelHyperparams, negLogEvidenceValue)
					else:
						print(negLogEvidence.counter, '\t', self.lengthScale, self.stationaryLengthScale, negLogEvidenceValue)

			return(negLogEvidenceValue)

		# VERBOSE
		if VERBOSE:
			negLogEvidence.lastPrintedSec = -1
			negLogEvidence.counter = 0

		if VERBOSE:
			if _optimiseLengthHyperparams:
				print('Format: (Iteration Counter) \t (--Length Scales--) (Stationary Length Scale) (--Length Scale Hyperparameters--) (negative-log-marginal-likelihood)')
			else:
				print('Format: (Iteration Counter) \t (--Length Scales--) (Stationary Length Scale) (negative-log-marginal-likelihood)')

		if theta == None:

			if _optimiseLengthHyperparams:
				thetaInitial = np.append(self.lengthScale.copy(), self.gpLengthScale.getKernelHyperparams())
				lowerBounds = np.append(self.minLengthScale * np.ones(self.lengthScale.shape[0]), np.array([1e-8, self.stationaryLengthScale]))
			else:
				thetaInitial = self.lengthScale.copy()
				lowerBounds = self.minLengthScale * np.ones(self.lengthScale.shape[0])

			# Initialise our optimiser with the right amount of parameters
			opt = nlopt.opt(nlopt.LN_COBYLA, thetaInitial.shape[0])

			# Set a non-zero but small lower bound for computational safety

			opt.set_lower_bounds(lowerBounds)

			# Set an appropriate tolerance level
			opt.set_ftol_rel(1e-2)
			opt.set_xtol_rel(1e-2)

			# Don't let the optimiser run too long
			opt.set_maxtime(maxtime)

			# We want to maximise the log-marginal-likelihood or log-evidence
			opt.set_min_objective(negLogEvidence)

			# Learn the hyper-parameters
			thetaFinal = opt.optimize(thetaInitial)

			for i in range(relearn):
				self.gpLengthScale.learn(sigma = 0, thetaLowerBound =  np.array([self.gpLengthScale.getKernelHyperparams()[0], self.hyperLengthScaleMin]))
				thetaFinal = opt.optimize(thetaFinal)

			if optimiseLengthHyperparams:
				self.learn(_optimiseLengthHyperparams = True)
				
			# Obtain the log-marginal-likelihood
			self.logMaginalLikelihood = -opt.last_optimum_value()

		else:

			self.logMaginalLikelihood = -negLogEvidence(theta, np.zeros(theta.shape))


		if VERBOSE:
			if _optimiseLengthHyperparams:
				print('Finished Learning: (--Length Scales--) (Stationary Length Scale) (--Length Scale Hyperparameters--) (log-marginal-likelihood)')
				print(self.lengthScale, self.stationaryLengthScale, self.gpLengthScale.kernelHyperparams, self.logMaginalLikelihood)
			else:
				print('Finished Learning: (--Length Scales--) (Stationary Length Scale) (log-marginal-likelihood)')
				print(self.lengthScale, self.stationaryLengthScale, self.logMaginalLikelihood)

		# Compute the gradient of the model for the sake of it
		self.updateGradient()

	def predict(self, Xq, predictUncertainty = True, predictObservations = False):

		# Compute the prediction using the standard Gaussian Process Prediction Algorithm
		alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
		Kq = self.kernel(self.X, Xq)
		fqExp = np.dot(Kq.T, alpha) + self.yMean

		# Compute the prediction covariance using the standard Gaussian Process Prediction Algorithm if requested
		if predictUncertainty:
			v = la.solve_triangular(self.L, Kq, lower = True, check_finite = False)
			Kqq = self.kernel(Xq, Xq)
			fqVar = (Kqq - np.dot(v.T, v))

		# Depending on if the user wants the one for latent function or output function, return the correct expected values and covariance matrix
		if predictUncertainty:
			if predictObservations:
				return(fqExp, fqVar + self.sigma**2 * np.eye(fqVar.shape[0]))
			else:
				return(fqExp, fqVar)
		else:
			return(fqExp)