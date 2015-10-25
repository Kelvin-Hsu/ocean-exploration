"""
Gaussian Process Classifier Prediction
Kelvin
"""
import numpy as np
import scipy.linalg as la
from computers.gp import compose
from computers.gp.linalg import jitchol
from computers.gp.partools import parallel_map, parallel_starmap

# Notes:
# For both the laplace approximation and probabilisitic least squares,
# a computational efficient way to compute
# the integral of a sigmoid multiplied by a gaussian pdf is the following
# This computation is exact for probit response and approximate for other
# responses (it is still very accurate)
# If really needed, we can use quadratures later for non probit responses
# This also means that for now, the PLS method does NOT utilise the 
# given response function

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Prediction: Module Classes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Predictor:
    """
    class 'Predictor'
    Description:
        Caches the training-query covariance matrix and query points
    Methods:
        None
    """
    def __init__(self, Xq, Kq):
        self.Xq = Xq
        self.Kq = Kq

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Prediction: Inference Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def query(memory, Xq, processes = 1):
    """
    Creates query predictor or list thereof for caching
    Arguments:
        memory(*)   :   Memory object learned from classifier learning
        Xq:         :   Query features
    Keyword Arguments:
        processes   :   Number of cores to use for parallelising computations
    Returns:
        predictor(*):   Predictor object cached with predictive covariance
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(memory, list):
        return parallel_map(query, memory, Xq, processes = processes)

    kernel = compose(memory.kerneldef)
    Kq = kernel(memory.X, Xq, memory.hyperparams)
    return Predictor(Xq, Kq)

def expectance(memory, predictor, processes = 1):
    """
    Computes predictive expectance of latent function
    Arguments:
        memory(*)   :   Memory object learned from classifier learning
        predictor(*):   Predictor object cached with predictive covariance
    Keyword Arguments:
        processes   :   Number of cores to use for parallelising computations
    Returns:
        fq_exp(*)   :   Predictive expectance at query points
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(memory, list):
        memories_predictors = [(memory[i], predictor[i]) 
            for i in range(len(memory))]
        return parallel_starmap(expectance, memories_predictors, 
            processes = processes)

    if memory.approxmethod == 'laplace':
        return np.dot(predictor.Kq.T, memory.cache.get('dloglik'))
    elif memory.approxmethod == 'pls':
        return np.dot(predictor.Kq.T, memory.cache.get('alpha'))

def variance(memory, predictor, processes = 1):
    """
    Computes predictive variance of latent function
    Arguments:
        memory(*)   :   Memory object learned from classifier learning
        predictor(*):   Predictor object cached with predictive covariance
    Keyword Arguments:
        processes   :   Number of cores to use for parallelising computations
    Returns:
        fq_var(*)   :   Predictive variance at query points
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(memory, list):
        memories_predictors = [(memory[i], predictor[i]) 
            for i in range(len(memory))]
        return parallel_starmap(variance, memories_predictors, 
            processes = processes)

    kernel = compose(memory.kerneldef)
    if memory.approxmethod == 'laplace':
        v = la.solve_triangular(memory.cache.get('L'), 
            (memory.cache.get('wsqrt') * predictor.Kq.T).T, 
            lower = True, check_finite = False)
    elif memory.approxmethod == 'pls':
        v = la.solve_triangular(memory.cache.get('L'), predictor.Kq,
            lower = True, check_finite = False)
    kqq = kernel(predictor.Xq, None, memory.hyperparams)
    fq_var = kqq - np.sum(v**2, axis = 0)
    return fq_var

def covariance(memory, predictor, processes = 1):
    """
    Computes predictive covariance of latent function
    Arguments:
        memory(*)   :   Memory object learned from classifier learning
        predictor(*):   Predictor object cached with predictive covariance
    Keyword Arguments:
        processes   :   Number of cores to use for parallelising computations
    Returns:
        fq_cov(*)   :   Predictive covariance at query points
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(memory, list):
        memories_predictors = [(memory[i], predictor[i]) 
            for i in range(len(memory))]
        return parallel_starmap(covariance, memories_predictors, 
            processes = processes)

    kernel = compose(memory.kerneldef)
    if memory.approxmethod == 'laplace':
        v = la.solve_triangular(memory.cache.get('L'), 
            (memory.cache.get('wsqrt') * predictor.Kq.T).T, 
            lower = True, check_finite = False)
    elif memory.approxmethod == 'pls':
        v = la.solve_triangular(memory.cache.get('L'), predictor.Kq,
            lower = True, check_finite = False)
    Kqq = kernel(predictor.Xq, predictor.Xq, memory.hyperparams)
    fq_cov = Kqq - np.dot(v.T, v)
    return fq_cov

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Prediction: Draw Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def latentdraws(n_draws, exp, covar, S = None, processes = 1):
    """
    Draws latent functions for each given binary classifier
    Arguments:
        n_draws     :   Number of functions to draw
        exp(*)      :   Expectance vector of a
                        finite collection of points on the latent function
        covar(*)    :   Covariance matrix or variance vector of a
                        finite collection of points on the latent function
    Keyword Arguments:
        S           :   Cached iid univariate Gaussian samples    
        processes   :   Number of cores to use for parallelising computations
    Returns:
        D           :   Matrix of latent function draws
                        Shape: (n_draws, n_query) where n_query = exp.shape[0]
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(exp, list):
        args = [(n_draws, exp[i], covar[i], S) for i in range(len(exp))]
        return parallel_starmap(latentdraws, args, processes = processes)

    if S == None:                       # S: Standard Draw (Cachable)
        S = np.random.normal(loc = 0., scale = 1., 
            size = (exp.shape[0], n_draws))

    if covar.ndim == 2:                 # 'covar' is a Covariance Matrix
        L = jitchol(covar,              # L: Covariance Cholesky Decomposition
            overwrite_a = True, check_finite = False)
        C = np.dot(L, S)                # C: Transform to include covariance
        D = (C.T + exp)                 # D: Transform to include expectance
    else:                               # 'covar' is a Variance Vector
        D = exp + S.T * np.sqrt(covar)  # D: Sample Independently
    return D

def draws(n_draws, exp, covar, memory, 
    S = None, return_indices = False, processes = 1):
    """
    Draws class labels from Gaussian process classifier
    Works for both binary and multiclass (OVA/AVA) classifier
    Arguments:
        n_draws     :   Number of functions to draw
        exp(*)      :   Expectance vector of a
                        finite collection of points on the latent function
        covar(*)    :   Covariance matrix or variance vector of a
                        finite collection of points on the latent function
        memory(*)   :   Memory object learned from classifier learning
    Keyword Arguments:
        S           :   Cached iid univariate Gaussian samples
        processes   :   Number of cores to use for parallelising computations
    Returns:
        y_draws     :   Matrix of class label draws
                        Shape: (n_draws, n_query) where n_query = exp.shape[0]
    (*) Accepts homogenous lists of described quantity
    """
    # f_draws.shape = (n_class, n_draws, n_query) for OVA
    # f_draws.shape = ((n_class * (n_class - 1) /2), n_draws, n_query) for AVA
    # f_draws.shape = (n_draws, n_query) for binary
    f_draws = np.array(latentdraws(n_draws, exp, covar, 
        S = S, processes = processes))

    if isinstance(memory, list):    # For multiclass classification...

        if memory[0].cache.get('multimethod') == 'OVA': # For OVA method...
            # Map to label with highest latent
            indices = f_draws.argmax(axis = 0) 

        elif memory[0].cache.get('multimethod') == 'AVA': # For AVA method...
            # Map to label with highest combined latent
            n_class = memory[0].cache.get('n_class')
            n_query = exp[0].shape[0]
            f_draws_hyper = np.zeros((n_class, n_class, n_draws, n_query))
            f_draws_hyper[memory[0].cache.get('upper_ind')] = +f_draws
            f_draws_hyper[memory[0].cache.get('lower_ind')] = -f_draws

            indices = f_draws_hyper.sum(axis = 0).argmax(axis = 0) 

        if return_indices:
            return indices

        return memory[0].cache.get('y_unique')[indices]

    else:                           # For binary classification...

        y_draws = np.ones(f_draws.shape)    # latent >= 0 means label +1
        if return_indices:
            y_draws[f_draws < 0] = 0        # latent <  0 means index 0
        else:
            y_draws[f_draws < 0] = -1       # latent <  0 means label -1
            
        return y_draws.astype('int64')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Prediction: Prediction Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def predict(Xq, memory, fusemethod = 'EXCLUSION'):
    """
    Wrapper function for classifier prediction for all cases
    Simply pass in the learned memory object for classifier learning
    
    Arguments:
        Xq:         :   Query features
        memory(*)   :   Memory object learned from classifier learning
    Keyword Arguments:
        fusemethod  :   Method of fusing probabilities for multiclass class
        processes   :   Number of cores to use for parallelising computations
    Returns:
        piq         :   Expected probability distribution of classes
    (*) Accepts homogenous lists of described quantity
    """
    predictor = query(memory, Xq)
    exp = expectance(memory, predictor)
    var = variance(memory, predictor)
    return predict_from_latent(exp, var, memory, fusemethod = fusemethod)

def predict_from_latent(exp, covar, memory, fusemethod = 'EXCLUSION'):
    """
    Wrapper function for classifier prediction for all cases
    Simply pass in the learned memory object for classifier learning
    
    Arguments:
        exp(*)      :   Expectance vector of a
                        finite collection of points on the latent function
        covar(*)    :   Covariance matrix or variance vector of a
                        finite collection of points on the latent function
        memory(*)   :   Memory object learned from classifier learning
    Keyword Arguments:
        fusemethod  :   Method of fusing probabilities for multiclass class
        processes   :   Number of cores to use for parallelising computations
    Returns:
        piq         :   Expected probability distribution of classes
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(memory, list):

        yq_probs = np.array([predict_from_latent(exp[i], covar[i], memory[i], 
            fusemethod = fusemethod) for i in range(len(memory))])

        multimethod = memory[0].cache.get('multimethod')

        if multimethod == 'OVA':

            # Fuse the probabilities using the specified fuse method
            return fuse_probabilities_OVA(yq_probs, fusemethod = fusemethod)

        elif multimethod == 'AVA':

            # Initialise the hyper-matrix of the expected probabilities
            n_class = memory[0].cache.get('n_class')
            nq = exp[0].shape[0]
            yq_prob_hyp = np.ones((n_class, n_class, nq))

            # Extract the expected AVA prediction probability of each class pair
            yq_prob_hyp[memory[0].cache.get('upper_ind')] = yq_probs
            yq_prob_hyp[memory[0].cache.get('lower_ind')] = 1 - yq_probs

            # Fuse the probabilities using the specified fuse method
            return fuse_probabilities_AVA(yq_prob_hyp, fusemethod = fusemethod)

    if covar.ndim == 2:         # 'covar' is a Covariance Matrix
        return memory.response.marginalise_latent(exp, covar.diagonal())
    else:                       # 'covar' is a Variance Vector
        return memory.response.marginalise_latent(exp, covar)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Prediction: Probability Fusion Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def fuse_probabilities_OVA(yq_prob, fusemethod = 'EXCLUSION'):
    """
    Fuses and normalises the predicted probabilities in multiclass OVA case
    Arguments:
        yq_prob     :   Original expected probability distribution of classes
                        Shape: (n_class x n_obs)
    Keyword Arguments:
        fusemethod  :   Method of fusing probabilities for multiclass class
    Returns:
        yq_prob     :   Fused expected probability distribution of classes
                        Shape: (n_class x n_obs)
    """
    # much weird, very hack, wow (but it works!)
    if fusemethod == 'MODE':        

        # Form the query index set
        Iq = np.arange(yq_prob.shape[1])

        # Find where the maximum probabilities occur for each query point
        yq_pred = yq_prob.argmax(axis = 0)

        # Find those maximum probabilities for each observation
        yq_prob_max = yq_prob[yq_pred, Iq]

        # The sum of the other probabilities is our normaliser
        normaliser = 1 - yq_prob_max

        # Normalise the probabilities so that the other probabilities 
        # sum up to the normaliser
        yq_prob *= (normaliser/(yq_prob.sum(axis = 0) - yq_prob_max))

        # Re-fill the maximum probabilities as we just changed it 
        # (change it back)
        yq_prob[yq_pred, Iq] = yq_prob_max

        # Return the normalised expected probability matrix
        return yq_prob

    elif fusemethod == 'EXCLUSION':

        # Find the number of observations we have
        n_class = yq_prob.shape[0]

        # Find the complement probabilities
        inv_yq_prob = 1 - yq_prob

        # In order to perform the next vectorised step, compute the correction
        # factor for each row now
        factor_yq_prob = yq_prob / inv_yq_prob

        # Multiply all the complement probabilities and correct each row with 
        # the correction factor
        new_yq_prob = np.repeat(np.array([inv_yq_prob.prod(axis = 0)]), 
            n_class, axis = 0) * factor_yq_prob

        # Normalise the probabilities
        yq_prob = new_yq_prob / new_yq_prob.sum(axis = 0)

        # Return the normalised expected probability matrix
        return yq_prob

    elif fusemethod == 'NORM':

        yq_prob /= yq_prob.sum(axis = 0)

        return yq_prob

    else:
        raise ValueError('There is no probability fusion method "%s" ', 
            fusemethod)

def fuse_probabilities_AVA(yq_prob_hyp, fusemethod = 'EXCLUSION'):
    """
    Fuses and normalises the predicted probabilities in multiclass AVA case
    Arguments:
        yq_prob_hyp :   Original expected probability distribution of classes
                        Shape: (n_class x n_class x n_obs)
    Keyword Arguments:
        fusemethod  :   Method of fusing probabilities for multiclass class
    Returns:
        yq_prob     :   Fused expected probability distribution of classes
                        Shape: (n_class x n_obs)
    """

    # Reduce the problam to an OVA problem by combining the
    # expected predictive probabilities across c' for classifiers between 
    # c and c'
    # This gives the predictive probabilities for each class c
    # For Mode Keeping, use the mean probability across c'
    # For all other methods, use the consistency probability across c'
    if fusemethod == 'MODE':
        yq_prob = yq_prob_hyp.mean(axis = 1)

    else:
        yq_prob = yq_prob_hyp.prod(axis = 1)
    return fuse_probabilities_OVA(yq_prob, fusemethod = fusemethod)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Prediction: Classification and Entropy Analysis Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def classify(yq_prob, y_ref):
    """ 
    Converts the probability distribution outputed by multiclass prediction 
    to the actual class prediction by finding the class corresponding with 
    the mode probability
    Arguments:
        yq_prob: The query probabilities from the predict method
        y_ref  : A reference 1D array of all the multiclass labels       
    Returns:
        yq_pred: The class label with mode probability
    """
    # Return the label corresponding to the mode of probability distribution
    y_unique = np.unique(y_ref)
    n_class = y_unique.shape[0]
    if n_class > 2:
        return y_unique[yq_prob.argmax(0)]
    elif n_class == 2:
        return y_unique[(yq_prob >= 0.5).astype(int)]
    else:
        raise ValueError('Number of classes is less than 2...')

def entropy(yq_prob):
    """
    Finds the entropy of the predicted probabilities from the classifiers
    This works for both binary and multiclass classification
    Arguments:
        yq_prob: The query probabilities from the predict method
    Returns:
        entropy: The entropy at each query point
    """
    # Binary and multiclass classifiers involve slightly different computations
    if yq_prob.ndim == 2:       # MULTICLASS GIVES MATRIX WHICH IS 2D
        entropy = - (yq_prob * np.log(yq_prob)).sum(0)
    else:                       # BINARY GIVES VECTOR WHICH IS 1D
        entropy = - (yq_prob * np.log(yq_prob)) - \
                    ((1 - yq_prob) * np.log(1 - yq_prob)) 
    return entropy

def gaussian_entropy(covar):
    """
    Obtains the joint entropy of a multivariate Gaussian distribution
    This is equivalent to obtaining the joint entropy of the latent function
    given the covariance matrix between given finite samples of said latent
    If only the variances are given, then a vector of marginalised entropy is 
    returned for each given point on the latent function instead
    Arguments:
        covar(*)    :   Covariance matrix or variance vector of a
                        finite collection of points on the latent function
    Keyword Arguments:
        processes   :   Number of cores to use for parallelising computations
    Returns:
        entropy     :   The joint entropy or vector of marginalised entropy
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(covar, list):
        return gaussian_entropy(np.array(covar).sum(axis = 0))

    const = 0.5 * np.log(2 * np.pi * np.e)
    if covar.ndim == 2:         # 'covar' is a Covariance Matrix
        L = jitchol(covar, overwrite_a = True, check_finite = False)
        return np.sum(np.log(L.diagonal())) + covar.shape[0] * const
    else:                       # 'covar' is a Variance Vector
        np.clip(covar, 1e-323, np.inf, out = covar)
        return 0.5 * np.log(covar) + const

def linearised_model_differential_entropy(exp, covar, memory):
    """
    Obtains a linearised estimate of the joint entropy of the class prediction 
    probabilities given the covariance matrix between given finite samples of 
    said latent function
    If only the variances are given, then a vector of marginalised entropy is 
    returned for each given point on the latent function instead
    Arguments:
        exp(*)      :   Expectance vector of a
                        finite collection of points on the latent function
        covar(*)    :   Covariance matrix or variance vector of a
                        finite collection of points on the latent function
        memory(*)   :   Memory object learned from classifier learning
    Returns:
        entropy     :   The joint entropy or vector of marginalised entropy
    (*) Accepts homogenous lists of described quantity
    """

    # For multiclass problems, we linearise the softmax
    if isinstance(memory, list):

        # Convert the AVA latent distribution to that of OVA if necessary
        if memory[0].cache.get('multimethod') == 'AVA':
            exp, covar = latent_AVA2OVA(exp, covar, memory)

        # Exponentiate the latent expectance
        exp_f = np.exp(np.array(exp))

        # Compute their sum and square thereof
        sum_exp_f = exp_f.sum(axis = 0)
        sum_exp_f_sq = sum_exp_f**2

        # Index each class in standard order
        classes = np.arange(memory[0].cache.get('n_class'))

        def covar_of_class(m):
            """
            Private function solely for 'linearised_entropy'
            Computes the linearised covariance matrix of the 
            prediction probability of the class indexed by 'm'
            """
            # Choose the (exponentiated) latent expectance of relevant class 
            exp_f_m = exp_f[m]

            # Compute the gradient of the softmax
            grad_softmax = - (exp_f_m / sum_exp_f_sq) * exp_f
            grad_softmax[m] += exp_f_m/sum_exp_f

            # Compute the linearised bi-linear coefficients
            if covar[0].ndim == 2:
                factor = np.array([np.outer(grad_softmax[k], grad_softmax[k]) 
                    for k in classes])
            else:
                factor = grad_softmax**2
                
            # Compute the linearised covariance matrix
            return (factor * np.array(covar)).sum(axis = 0)

        # Compute the linearised entropy
        return gaussian_entropy([covar_of_class(m) for m in classes])

    # For binary problems, we linearise the sigmoid
    grad_exp = memory.response.gradient(exp)
    if covar.ndim == 2:         # 'covar' is a Covariance Matrix
        return gaussian_entropy(np.outer(grad_exp, grad_exp) * covar)
    else:                       # 'covar' is a Variance Vector
        return gaussian_entropy(grad_exp**2 * covar)

def monte_carlo_prediction_information_entropy(exp, covar, memory, 
    n_draws = 1000, S = None, vectorise = True, processes = 1):
    """
    Computes the joint entropy of a query region using monte carlo sampling
    Works for both binary and multiclass classifiers
    Arguments:
        exp(*)      :   Expectance vector of a
                        finite collection of points on the latent function
        covar(*)    :   Covariance matrix or variance vector of a
                        finite collection of points on the latent function
        memory(*)   :   Memory object learned from classifier learning
    Keyword Arguments:
        n_draws     :   Number of functions to draw
        S           :   Cached iid univariate Gaussian samples
        vectorise   :   Only applies when estimating marginalised entropies
                        When vectorised, bottleneck iterations are over the
                        number of classes instead of the number of query 
                        points. The downside is that this will use up much more
                        memory. Be careful!
        processes   :   Number of cores to use for parallelising computations
    Returns:
        entropy     :   Estimated joint entropy of query region [scalar]
    (*) Accepts homogenous lists of described quantity
    """

    # Sample from the gp classifier
    s = draws(n_draws, exp, covar, memory, 
        S = S, return_indices = True, processes = processes)

    # If jointly sampled...
    if (isinstance(covar, list) and covar[0].ndim == 2) or \
        (not isinstance(covar, list) and covar.ndim == 2):

        # Find the unique rows and frequencies of occurance
        all_freq = (s[:, np.newaxis, :] == s).all(axis = 2).sum(axis = 1)
        _, idx = np.unique(np.ascontiguousarray(s).view(np.dtype((np.void, 
            s.dtype.itemsize * s.shape[1]))), return_index = True)
        
        # Determine the frequencies of occurance of each distinct observation
        # Undrawed realisations will not appear nor contribute to entropy anyway
        # The order does not matter here
        p = all_freq[idx].astype(float) / n_draws

        # Compute the entropy from the estimated joint probabilitiess
        return - (p * np.log(p)).sum()

    # If independently sampled...
    # Both methods above will compute the exact same thing, but with
    # different time and memory requirements
    else:

        # Compute entropy through iterating through classes
        if vectorise:

            classes = np.arange(2 if not isinstance(exp, list) else len(exp))
            def h(p):
                return - (p * np.log(np.clip(p, 1e-323, np.inf))).sum(0)
            return h(np.array([(s == c).mean(axis = 0) for c in classes]))

        # Compute entropy separately for each query point
        else:

            queries = np.arange(s.shape[1])
            def H(p):
                p = p[p != 0]
                return - (p * np.log(p)).sum()
            return np.array([H(np.bincount(s[:, i])/n_draws) for i in queries])



def equivalent_standard_deviation(entropy):
    """
    Computes the equivalent standard deviation from an entropy measure
    It is possible that in some cases the actual equivalent standard deviation
    is a constant multiple of the returned value
    This function is written more for visualisation purposes (the heat map 
    will be correct)
    Arguments:
        entropy     :   Entropy of a Univariate Gaussian or arrays thereof
    Returns:
        sd_equiv    :   Equivalent standard deviation
    """
    return np.exp(entropy - 0.5 * np.log(2 * np.pi * np.e))
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Prediction: Miscellaneous Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def latent_AVA2OVA(exp, covar, memory):
    """
    Fuses latent expectances and covariances/variances for the AVA case into 
    an equivalent OVA format. This is used by 'linearised_entropy'.
    Arguments:
        exp(*)      :   List of AVA expectance vector
        covar(*)    :   List of AVA covariance matrix or variance vector
        memory(*)   :   Memory object learned from AVA classifier learning
    Returns:
        exp(*)      :   List of OVA expectance vector
        covar(*)    :   List of OVA covariance matrix or variance vector
    (*) Only supports list types described above
    """
    n_class = memory[0].cache.get('n_class')

    exp_table = np.zeros((n_class, n_class, exp[0].shape[0]))
    covar_table = np.zeros((n_class, n_class) + covar[0].shape)

    upper_ind = memory[0].cache.get('upper_ind')
    lower_ind = memory[0].cache.get('lower_ind')

    exp_table[upper_ind] = np.array(exp)
    exp_table[lower_ind] = -np.array(exp)
    covar_table[upper_ind] = np.array(covar)
    covar_table[lower_ind] = np.array(covar)

    exp_fused = list(exp_table.sum(axis = 1))
    covar_fused = list(covar_table.sum(axis = 1))

    return exp_fused, covar_fused

def cov2var(cov):
    """
    Extracts variance vectors from the covariance matrix
    Arguments:
        cov(*)      :   Covariance matrix of a
                        finite collection of points on the latent function
    Returns:
        var(*)      :   Variance matrix of a
                        finite collection of points on the latent function
    """
    if isinstance(cov, list):
        return [c.diagonal() for c in cov]
    return cov.diagonal()

