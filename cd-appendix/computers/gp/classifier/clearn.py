"""
Gaussian Process Classifier Learning and Training

Kelvin
"""
import numpy as np
import scipy.linalg as la
from computers.gp import compose
from computers.gp.linalg import jitchol
from computers.gp.train import optimise_hypers
from computers.gp.partools import parallel_map, parmap

import copy
import time
import logging
from scipy.sparse import identity

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Learning: Module Classes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Memory:
    """
    class 'Memory'

    Description:
        Stores the learned memory and cache for each binary classifier

    Methods:
        None
    """
    def __init__(self, approxmethod, X, y, kerneldef, response, hyperparams, 
        logmarglik):
        self.approxmethod = approxmethod
        self.X = X
        self.y = y
        self.kerneldef = kerneldef
        self.response = response
        self.hyperparams = hyperparams
        self.logmarglik = logmarglik
        self.cache = {}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Learning: Wrapper Methods for Learning
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def learn(X, ym, kerneldef, response, optconfig, 
    multimethod = 'OVA', processes = 1, **kwargs):
    """
    Determines the correct learning method to be used given learning options
    
    Arguments:
        X        : Training Features
        ym       : Training Labels
        kerneldef: Kernel Covariance Function Definition
        response : Response Sigmoid Function (as provided in 'responses.py')
        optconfig: An instance of the 'OptConfig' class (see 'types.py')
    Keyword Arguments:
        multimethod : Multiclass classification method
        approxmethod: Approximation method for satisfying the GP model
        train       : To optimise the latent function learning or not
        ftol        : Tolerance level for convergence
        processes   : Number of cores to use for parallel learning
    Returns:
        memories    : Learned Memory and Cache for Prediction
    """
    # Determine the unique labels and the number of classes
    ym_unique = np.unique(ym)
    n_class = ym_unique.shape[0]

    kernel = compose(kerneldef)

    # Determine if it is a binary or multiclass classification problem
    if n_class == 2:

        y = labels2indicators(ym) # Ensure they are -1 and 1 indicators
        memory = learn_binary(X, y, kerneldef, response, optconfig, **kwargs)
        return memory

    elif n_class > 2:

        if multimethod == 'AVA':

            memories = learn_multiclass_AVA(X, ym, kerneldef, 
                response, optconfig, processes = processes, **kwargs)

        elif multimethod == 'OVA':

            memories = learn_multiclass_OVA(X, ym, kerneldef, 
                response, optconfig, processes = processes, **kwargs)

        else:
            raise TypeError('No multiclass classification method "%s"'
                % multimethod)

        return memories

    else:
        raise TypeError('Number of classes is less than 2...')

def learn_binary(X, y, kerneldef, response, optconfig, 
    approxmethod = 'laplace', **kwargs):
    """
    Chooses the correct binary classifier for learning
    
    Arguments:
        X        : Training Features
        ym       : Training Labels
        kerneldef: Kernel Covariance Function Definition
        response : Response Sigmoid Function (as provided in 'responses.py')
        optconfig: An instance of the 'OptConfig' class (see 'types.py')
    Keyword Arguments:
        approxmethod: Approximation method for satisfying the GP model
        train       : To optimise the latent function learning or not
        ftol        : Tolerance level for convergence
    Returns:
        memories    : Learned Memory and Cache for Prediction
    """
    laplace_strings = ['laplace', 'LAPLACE', 'Laplace']
    ep_strings = ['ep', 'EP', 'Expectation Propagation']
    pls_strings = ['pls', 'PLS', 'Probabilistic Least Squares']

    if approxmethod in laplace_strings:
        memory = learn_binary_laplace(X, y, kerneldef, response, optconfig, 
                                        **kwargs)
    elif approxmethod in pls_strings:
        memory = learn_binary_pls(X, y, kerneldef, response, optconfig, 
                                        **kwargs)
    elif approxmethod in ep_strings:
        memory = learn_binary_ep(X, y, kerneldef, response, optconfig, 
                                        **kwargs)
    return memory
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Learning: Binary Classifier Learning Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def learn_binary_laplace(X, y, kerneldef, response, optconfig, 
    train = True, ftol = 1e-10, tasklabel = None):
    """ 
    Binary classification learning with laplace approximation

    Arguments:
        X        : Training Features
        y        : Training Labels
        kerneldef: Kernel Covariance Function Definition
        response : Response Sigmoid Function (as provided in 'responses.py')
        optconfig: An instance of the 'OptConfig' class (see 'types.py')
    Keyword Arguments:
        train    : To optimise the latent function learning or not
        ftol     : The tolerance level for convergence when training the 
                   latent function
        tasklabel: Task label for logging purposes
    Returns:
        memory   : A cache object for quantities used in prediction
    """
    # Compose the kernel and prepare quantities for learning
    kernel = compose(kerneldef)
    (n, k) = X.shape
    I = identity(n)

    # This is the closure to find the negative-log-marginal-likelihood
    def nlml(theta, noise):

        # Use Newton-Raphson Method to learn the latent function
        K = kernel(X, X, theta)
        fprev = nlml.f + 1e2*ftol
        while np.max(np.abs(nlml.f - fprev)) > ftol:

            # Note the current latent function as the previous latent function
            fprev = nlml.f.copy()

            # Compute the derivative and negative hessian of the log-likelihood
            nlml.dloglik = response.gradient_log_lik(y, nlml.f)
            nlml.w = -response.hessian_log_lik_diag(y, nlml.f)
            nlml.wsqrt = np.sqrt(nlml.w)

            # Re-learn the latent function
            nlml.L = jitchol(I + np.outer(nlml.wsqrt, nlml.wsqrt) * K, 
                overwrite_a = True, check_finite = False)
            b = nlml.w * nlml.f + nlml.dloglik
            c = la.solve_triangular(nlml.L, nlml.wsqrt * K.dot(b), 
                lower = True, overwrite_b = True, check_finite = False)
            a = b - nlml.wsqrt * la.solve_triangular(nlml.L.T, c, 
                overwrite_b = True, check_finite = False)
            nlml.f = K.dot(a)

        # Compute the neg-log-marginal-likelihood of the learning result
        nlmlvalue = -(-0.5 * a.dot(nlml.f) + response.log_lik(y, nlml.f) - \
            np.sum(np.log(nlml.L.diagonal())))

        # Log the current optimisation status
        logging.debug(  '\t\tTask Label: {0} | '
                        'Hyperparameters: {1} | '
                        'Log-Marginal-Likelihood: {2:.3f}'.format(
                            tasklabel, theta, -nlmlvalue))

        # Return the neg-log-marginal-likelihood
        return nlmlvalue

    # Initialise important quantities we can cache for prediction
    nlml.f = np.zeros(n)
    nlml.w = np.ones(n)
    nlml.wsqrt = np.ones(n)
    nlml.L = I.copy()
    nlml.dloglik = np.zeros(n)

    # To train or not to train
    logging.info('Initiating learning for task label {0} '
        'with hyperparameters {1} and log-marginal-likelihood: {2:.3f}'.format(
            tasklabel, optconfig.sigma.initialVal, 
            -nlml(optconfig.sigma.initialVal, None)))
    starttime = time.clock()
    if train:
        (theta, _, nlmlvalue) = optimise_hypers(nlml, optconfig)
    else:
        theta = optconfig.sigma.initialVal.copy()
        nlmlvalue = nlml(theta, None)
    learntime = time.clock() - starttime
    logging.info('Learning for task label {0} completed in %.4f seconds ' \
        'with hyperparameters {1} and log-marginal-likelihood: {2:.3f}'.format(
            tasklabel, theta, -nlmlvalue) % learntime)

    # Cache the result
    memory = Memory('laplace', X, y, kerneldef, response, theta, -nlmlvalue)
    memory.cache.update(
        {   'f': nlml.f,
            'w': nlml.w,
            'wsqrt': nlml.wsqrt,
            'L': nlml.L,
            'dloglik': nlml.dloglik     })
    memory.cache.update({'tasklabel': tasklabel, 'learntime': learntime})
    memory.cache.update({'multimethod': 'binary'})
    memory.cache.update({'n_class': 2})
    memory.cache.update({'y_unique': np.unique(y)})
    return memory

def learn_binary_pls(X, y, kerneldef, response, optconfig, 
    train = True, ftol = 1e-10, tasklabel = None):
    """ 
    Binary classification learning with probabilistic least squares

    Arguments:
        X        : Training Features
        y        : Training Labels
        kerneldef: Kernel Covariance Function Definition
        response : Response Sigmoid Function (as provided in 'responses.py')
        optconfig: An instance of the 'OptConfig' class (see 'types.py')
    Keyword Arguments:
        train    : To optimise the latent function learning or not
        ftol     : The tolerance level for convergence when training the 
                   latent function
        tasklabel: Task label for logging purposes
    Returns:
        memory   : A cache object for quantities used in prediction
    """
    from computers.gp import learn as regressor_learn
    from computers.gp import Range
    from computers.gp import condition

    # Compose the kernel
    kernel = compose(kerneldef)

    # If the user did not specify the noise level, use the following default
    if not optconfig.noise.initialVal:
        optconfig.noise = Range([1e-8], [2], [1])

    # Learn the classifier as a regressor
    logging.info('Initiating learning for task label {0} '
        'with hyperparameters {1}'.format(tasklabel, 
            optconfig.sigma.initialVal))
    starttime = time.clock()
    (hyperparams, noise, logmarglik) = \
        regressor_learn(X, y, kernel, optconfig, 
        optCrition = 'logMarg', returnLogMarg = True, verbose = False)
    learntime = time.clock() - starttime
    logging.info('Learning for task label {0} completed in %.4f seconds ' \
        'with hyperparameters {1} and log-marginal-likelihood: {2:.3f}'.format(
            tasklabel, hyperparams, logmarglik) % learntime)

    # Cache the regressor as well
    regressor = condition(X, y, kernel, (hyperparams, noise))

    # Cache the result
    memory = Memory('pls', X, y, kerneldef, response, hyperparams, logmarglik)
    memory.cache.update({   'L': regressor.L,
                            'alpha': regressor.alpha,
                            'noise': noise})
    memory.cache.update({'tasklabel': tasklabel, 'learntime': learntime})
    memory.cache.update({'multimethod': 'binary'})
    memory.cache.update({'n_class': 2})
    memory.cache.update({'y_unique': np.unique(y)})
    return memory

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Learning: Multiclass Classifier Learning Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def __learn_binary_OVA__(data, X, kerneldef, response,
    approxmethod, train, ftol):
    """
    Redefinition of 'learn_binary' for the OVA case for parallelisation
    Extracts iterable data and passes it to each binary classifier instance
    See 'learn_binary' for more documentation
    """
    logging.basicConfig(level = 
        parmap.multiprocessing.get_logger().getEffectiveLevel())
    y = data[0]
    optconfig = data[1]
    tasklabel = data[2]
    return learn_binary(X, y, kerneldef, response, optconfig, 
        approxmethod = approxmethod, train = train, ftol = ftol, 
        tasklabel = tasklabel)

def learn_multiclass_OVA(X, ym, kerneldef, response, optconfig, 
    approxmethod = 'laplace', train = True, ftol = 1e-10, processes = 1):
    """
    Learns latent functions for the multiclass classifier by learning 
    'n_class' binary classifiers for "one v.s. all" classification

    'n_class' is the number of classes

    Note: This code will purposely break through the outputs of 
    'labels2indicators' if the labels only contain 2 classes

    The following interface and usage is basically the same as 
    'learn_binary', except that the output labels 'ym' can have
    multiple labels (it doesn't matter what values they hold)

    Arguments:
        X        : Training Features
        ym       : Training Labels (multiclass)
        kerneldef: Kernel Covariance Function Definition
        response : Response Sigmoid Function (as provided in 'responses.py')
        optconfig: An instance of the 'OptConfig' class (see 'types.py')
    Keyword Arguments:
        approxmethod: Approximation method for satisfying the GP model
        train       : To optimise the latent function learning or not
        ftol        : Tolerance level for convergence
        processes   : Number of cores to use for parallel learning
    Returns:
        memories    : Learned Memory and Cache for Prediction
    """
    # Obtain indicator labels and the number of classes we have
    Y = labels2indicators(ym)
    (n_class, n_obs) = Y.shape
    memories = []
    data = []

    # Make sure the number of initial values is correct
    if isinstance(optconfig, list):
        if len(optconfig) != n_class:
            raise ValueError('Incorrect number of initial values')

    # Obtain iterable data to dispatch
    for i_class in range(n_class):
        y = Y[i_class]
        if isinstance(optconfig, list):
            data.append([y, optconfig[i_class], i_class])
        else:
            data.append([y, optconfig, i_class])

    # Parallise the learning stage
    start_time = time.clock()
    map_result = parallel_map(__learn_binary_OVA__, data, X, kerneldef,
                           response, approxmethod, train, ftol,
                           processes = processes)
    logging.info('Learning Time: %f' % (time.clock() - start_time))

    # Extract each learned memory
    y_unique = np.unique(ym)
    for i_class in range(n_class):
        memory = map_result[i_class]
        memory.cache.update({'i_class': i_class})
        memory.cache.update({'j_class': -1})
        memory.cache.update({'n_class': n_class})
        memory.cache.update({'y_unique': y_unique})
        memory.cache.update({'multimethod': 'OVA'})
        memories.append(memory)
    
    # Return the learned memory
    return memories

def __learn_binary_AVA__(data, kerneldef, response, 
    approxmethod, train, ftol):
    """
    Redefinition of 'learn_binary' for the AVA case for parallelisation
    Extracts iterable data and passes it to each binary classifier instance
    See 'learn_binary' for more documentation
    """
    logging.basicConfig(level = 
        parmap.multiprocessing.get_logger().getEffectiveLevel())
    Xs = data[0]
    ys = data[1]
    optconfig = data[2]
    tasklabel = data[3:]
    return learn_binary(Xs, ys, kerneldef, response, optconfig, 
        approxmethod = approxmethod, train = train, ftol = ftol, 
        tasklabel = tasklabel)

def learn_multiclass_AVA(X, ym, kerneldef, response, optconfig, 
    approxmethod = 'laplace', train = True, ftol = 1e-10, processes = 1):
    """
    Learns latent functions for the multiclass classifier by learning 
    'n_class*(n_class - 1)/2' binary classifiers for "all v.s. all" 
    classification

    'n_class' is the number of classes

    Note: This code will purposely break through the outputs of 
    'labels2indicators' if the labels only contain 2 classes

    The following interface and usage is basically the same as 
    'learn_binary', except that the output labels 'ym' can have
    multiple labels (it doesn't matter what values they hold)

    Arguments:
        X        : Training Features
        ym       : Training Labels (multiclass)
        kerneldef: Kernel Covariance Function Definition
        response : Response Sigmoid Function (as provided in 'responses.py')
        optconfig: An instance of the 'OptConfig' class (see 'types.py')
    Keyword Arguments:
        approxmethod: Approximation method for satisfying the GP model
        train       : To optimise the latent function learning or not
        ftol        : Tolerance level for convergence
        processes   : Number of cores to use for parallel learning
    Returns:
        memories    : Learned Memory and Cache for Prediction
    """

    # Obtain indicator labels and the number of classes we have
    Y = labels2indicators(ym)
    (n_class, n_obs) = Y.shape
    memories = []
    data = []

    # Make sure the number of initial values is correct
    if isinstance(optconfig, list):
        if len(optconfig) != (n_class * (n_class - 1) / 2):
            raise ValueError('Incorrect number of initial values')
        i = 0

    # Go through each unique combination of classes
    for i_class in range(n_class):
        for j_class in range(i_class + 1, n_class):

            # Obtain the corresponding indicators for each class
            y1 = Y[i_class]
            y2 = Y[j_class]

            # Obtain relevant training points for the two classes
            i_obs = (y1 == 1) | (y2 == 1)

            # Extract subset of relevant training features and indicator labels
            Xs = X[i_obs]
            ys = np.ones(i_obs.sum())
            ys[y2[i_obs] == 1] = -1

            # Zip the data together to pass as iterables later
            if isinstance(optconfig, list):
                data.append([Xs, ys, optconfig[i], i_class, j_class])
                i += 1
            else:
                data.append([Xs, ys, optconfig, i_class, j_class])

    # Parallise the learning stage
    start_time = time.clock()
    map_result = parallel_map(__learn_binary_AVA__, data, kerneldef, 
        response, approxmethod, train, ftol, processes = processes)
    logging.info('Learning Time: %f' % (time.clock() - start_time))

    # Extract each learned memory
    i_result = 0
    y_unique = np.unique(ym)
    upper_ind = np.triu_indices(n_class, k = 1)
    lower_ind = np.tril_indices(n_class, k = -1)
    for i_class in range(n_class):
        for j_class in range(i_class + 1, n_class):
            memory = map_result[i_result]
            memory.cache.update({'i_class': i_class})
            memory.cache.update({'j_class': j_class})
            memory.cache.update({'n_class': n_class})
            memory.cache.update({'y_unique': y_unique})
            memory.cache.update({'multimethod': 'AVA'})
            memory.cache.update({'upper_ind': upper_ind})
            memory.cache.update({'lower_ind': lower_ind})
            memories.append(memory)
            i_result += 1
    
    # Return the learned memory
    return memories

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classifier Learning: Other Utilities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def labels2indicators(y):
    """
    Transforms an array of labels to a matrix of indicators

    If there are only 2 labels involved, no matter what numerical values they 
    are, they will be converted into 1s and -1s as a 1D array
    Otherwise, for multiple labels, a matrix Y of size (n_class x n_obs) 
    is returned, where Y[i_class] is a 1D array of 1 and -1 indicators for 
    each class

    Arguments:
        y:  1D array of labels
    Returns:
        Y:  Matrix of binary 1 or -1 indicators
    """
    # Find the (sorted) unique labels in y and also the number of classes
    y_unique = np.unique(y)
    n_class = y_unique.shape[0]

    # Make sure it is in binary indicator form if problem if binary
    if n_class == 2:
        y_mean = y.mean()
        z = y.copy()
        z[y > y_mean] = 1
        z[y < y_mean] = -1
        return z

    # Compose the indicator matrix Y
    n_obs = y.shape[0]
    Y = -np.ones((n_class, n_obs))
    for i_class in range(n_class):
        Y[i_class][y == y_unique[i_class]] = 1

    return Y

def set_multiclass_logging_level(level):
    """
    Sets the logging level for tasks operating in parallel on different cores
    Arguments:
        level:  logging levels from the logging module
    ReturnsL
        None
    """
    parmap.multiprocessing.get_logger().setLevel(level)

def batch_start(opt_config, memory):
    """
    Sets initial values of the optimiser parameters
    Returned as an OptConfig instance or a list of OptConfig instances

    Arguments:
        opt_config      : An instance of OptConfig
        memory(*)       : Memory object learned from classifier learning
    Returns:
        batch_config(*) : Batched 'OptConfig' instance
    (*) Accepts homogenous lists of described quantity
    """
    if isinstance(memory, list):
        batch_config = []
        for bmemory in memory:
            opt_config_copy = copy.deepcopy(opt_config)
            opt_config_copy.sigma.initialVal = bmemory.hyperparams
            batch_config.append(opt_config_copy)
    else:
        batch_config = copy.deepcopy(opt_config)
        batch_config.sigma.initialVal = memory.hyperparams
    return batch_config