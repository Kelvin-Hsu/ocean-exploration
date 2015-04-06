import numpy as np
from scipy import stats

def logistic(z):

	response = 1/(1 + np.exp(-z))

	return(response)

def likelihoodLogistic(y, f):

	likelihood = 1/(1 + np.exp(-np.dot(y, f)))

	return(likelihood)

def logLikelihoodLogistic(y, f):

	logLikelihood = -np.log(1 + np.exp(-np.dot(y, f)))

	return(logLikelihood)

def gradientLogLikelihoodLogistic(y, f):

	gradient = (y + 1)/2 - logistic(f)

	return(gradient)

def hessianLogLikelihoodLogistic(y, f):

	pi = logistic(f)

	hessian = -np.diag(pi * (1 - pi))

	return(hessian)

logistic.likelihood = likelihoodLogistic
logistic.logLikelihood = logLikelihoodLogistic
logistic.gradientLogLikelihood = gradientLogLikelihoodLogistic
logistic.hessianLogLikelihood = hessianLogLikelihoodLogistic


def normalcdf(z):

	return(stats.norm.cdf(z))

def likelihoodNormalCdf(y, f):

	return(stats.norm.cdf(np.dot(y, f)))

def logLikelihoodNormalCdf(y, f):

	return(np.log(stats.norm.cdf(np.dot(y, f))))

def gradientLogLikelihoodNormalCdf(y, f):

	gradient = (y * stats.norm.pdf(f)) / stats.norm.cdf(y * f)

	return(gradient)

def hessianLogLikelihoodNormalCdf(y, f):

	pdf_f = stats.norm.pdf(f)
	yf = y * f
	cdf_yf = stats.norm.cdf(yf)
	hessian_diagonal = - (pdf_f / cdf_yf)**2 - (yf * pdf_f / cdf_yf)

	hessian = np.diag(hessian_diagonal)
	return(hessian)

normalcdf.likelihood = likelihoodNormalCdf
normalcdf.logLikelihood = logLikelihoodNormalCdf
normalcdf.gradientLogLikelihood = gradientLogLikelihoodNormalCdf
normalcdf.hessianLogLikelihood = hessianLogLikelihoodNormalCdf
