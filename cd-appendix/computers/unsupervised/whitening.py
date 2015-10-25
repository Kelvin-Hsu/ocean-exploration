""" A collections of various transformations that can applied to data,
    including PCA-whitening, standardisation, etc.

    Authors:    Daniel Steinberg, Alistair Reid
    Date:       March 2015
    Institute:  NICTA

"""


import numpy as np
import scipy.linalg as la


def standardise(X, params = None, inplace = False):
    """ Standardise the data, X. I.e. subtract the means of each columns of X,
        and divide by the columns standard deviations.

        Arguments:
            X: an [NxD] array.
            inplace: do this in-place, i.e. modify X.

        Returns:
            An [NxD] array of standardised data, unless inplace == True, in
            which case there is only the second return.
            A Tuple containing Xmean and Xstd for use with destandardise
    """

    if type(X) is list: # multi-task
        return [list(a) for a in zip(*[standardise(x, 
                params = params, inplace = inplace) for x in X])]

    if params is None:
        Xmean = X.mean(axis = 0)
        Xstd = X.std(axis = 0)
    else:
        Xmean = params[0]
        Xstd = params[1]

    if inplace:
        X[:] -= Xmean
        X[:] /= Xstd
        if params is None:
            return (Xmean, Xstd)
        else:
            return
    else:
        Xs = X - Xmean
        Xs /= Xstd
        if params is None:
            return Xs, (Xmean, Xstd)
        else:
            return Xs

def destandardise(X, params):
    if type(X) is list:
        return [ params[i][1]*X[i] + params[i][0] for i in range(len(X))]
    else:
        return X*params[1] + params[0]

def scaleVar(X, params):
    if type(X) is list:
        return [ params[i][1]**2 * X[i] for i in range(len(X))]
    else:
        return X*params[1]**2



def whiten(X, params=None, reducedims=None):
    """ Whiten input X using PCA whitening. This de-correlates and standardises
        multivariate X.

        This Works for the usual case of N > D, but also works if D > N, with
        aautomatic dimensionality reduction to min(N, D).

        Arguments:
            X: an [NxD] array.
            reducedims: integer value of the dimensionality of the returned
                whitened X, None for no dimensionality reduction.

        Return:
            Xw: an NxD whitened version of X, or Nxreducedims if not None.
            U: a DxD or Dxreducedims PCA projection matrix
            l: a vector of length D or reducedims of the singular values.
    """

    if type(X) is list: # multi-task
        if params is None:
            fullX = np.concatenate(X, axis=0)
            _, params = whiten(fullX, reducedims=reducedims)
            U, l, Xmean = params
            Xw = [(x-Xmean).dot(U / np.sqrt(l)) for x in X]
            return Xw, params
        else:
            U, l, Xmean = params
            return [(x-Xmean).dot(U / np.sqrt(l)) for x in X]

    N, D = X.shape

    if params is None:
        Xmean = X.mean(axis=0)
        X = X - Xmean
        S = X.T.dot(X) / (N-1) if N >= D else X.dot(X.T) / (N-1)
        U, l, _ = la.svd(S)
        if reducedims is not None:
            U = U[:, 0:reducedims]
            l = l[0:reducedims]
        if D > N:
            U = X.T.dot(U) / np.sqrt(l * (N-1))
    else:
        U, l, Xmean = params
        X = X - Xmean

    if params is None:
        return X.dot(U / np.sqrt(l)), (U, l, Xmean)
    else:
        return X.dot(U / np.sqrt(l))


def ApplyWhiten(X, U, l, xmean):
    """ Apply a pre-computed whitening transform to data.

        TODO

        Arguments:
            X:
            U:
            l:
            xmean:

        Returns:
    """

    return (X - xmean).dot(U / np.sqrt(l))
