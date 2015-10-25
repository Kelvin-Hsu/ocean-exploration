"""
module 'responses'

---Outline---
This module contains some reponse (sigmoid) functions
They are mainly used in Gaussian Processs Classifications

---Description---
Response functions are functions with univariate inputs and outputs
The likelihood of data y given model f in the classification case is
	p(y|f) := response(y dot f)
Here the response function itself is a struct containing its corresponding
likelihood, log likelihood, log likelihood gradient, and log likelihood hessian
functions. This is done for easy reference.
"""

import numpy as np
from scipy import stats

def probit(z):
	return stats.norm.cdf(z) 

def __gradient_probit__(z):
	return stats.norm.pdf(z)

def __lik_probit__(y, f):
	return np.prod(stats.norm.cdf(y * f))

def __log_lik_probit__(y, f):
	return np.sum(np.log(stats.norm.cdf(y * f)))

def __gradient_log_lik_probit__(y, f):
	return (y * stats.norm.pdf(f)) / stats.norm.cdf(y * f)

def __hessian_log_lik_diag_probit__(y, f):
	pdf_f = stats.norm.pdf(f)
	yf = y * f
	cdf_yf = stats.norm.cdf(yf)
	hessian_diagonal = - (pdf_f / cdf_yf)**2 - (yf * pdf_f / cdf_yf)
	return hessian_diagonal

def __hessian_log_lik_probit__(y, f):
	return np.diag(__hessian_log_lik_diag_probit__(y, f))

def __marginalise_latent_probit__(exp, var):
	return stats.norm.cdf(exp / np.sqrt(1 + var))

probit.gradient = __gradient_probit__
probit.lik = __lik_probit__
probit.log_lik = __log_lik_probit__
probit.gradient_log_lik = __gradient_log_lik_probit__
probit.hessian_log_lik = __hessian_log_lik_probit__
probit.hessian_log_lik_diag = __hessian_log_lik_diag_probit__
probit.marginalise_latent = __marginalise_latent_probit__
probit.name = 'probit'

def logistic(z):
	return 1/(1 + np.exp(-z)) 

def __gradient_logistic__(z):
	exp_z = np.exp(z)
	return exp_z / (1 + exp_z)**2

def __lik_logistic__(y, f):
	return 1/np.prod(1 + np.exp(-y * f))

def __log_lik_logistic__(y, f):
	return -np.sum(np.log(1 + np.exp(-y * f)))

def __gradient_log_lik_logistic__(y, f):
	return (y + 1)/2 - logistic(f)

def __hessian_log_lik_diag_logistic__(y, f):
	pi = logistic(f)
	return -pi * (1 - pi)

def __hessian_log_lik_logistic__(y, f):
	return np.diag(__hessian_log_lik_diag_logistic__(y, f))

def __marginalise_latent_logistic__(exp, var):
	return stats.norm.cdf(exp / np.sqrt(1 + var))

logistic.gradient = __gradient_logistic__
logistic.lik = __lik_logistic__
logistic.log_lik = __log_lik_logistic__
logistic.gradient_log_lik = __gradient_log_lik_logistic__
logistic.hessian_log_lik = __hessian_log_lik_logistic__
logistic.hessian_log_lik_diag = __hessian_log_lik_diag_logistic__
logistic.marginalise_latent = __marginalise_latent_logistic__
logistic.name = 'logistic'

names = ['probit', 'logistic']
def get(responsename):
	return eval(responsename) if responsename in names else None

# from scipy import integrate
# def __integrand__(z, func, exp, var):
# 	func(z) * np.exp(-(z - exp)**2/(2*var)) / np.sqrt(2 * np.pi * var)

# def __integrate_with_gaussian__(func, exp, var):

# 	inf = np.inf * np.ones(exp.shape)
# 	return integrate.quadrature(integrand, -inf, inf, args = (func, exp, var))

import sys
import inspect
functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)