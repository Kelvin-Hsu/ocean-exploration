"""
Demonstration of simple exploration algorithms using generated data

Author: Kelvin
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from computers import gp
import computers.unsupervised.whitening as pre
import sys
import logging
import shutil
import nlopt
import time
import parmap
from kdef import kerneldef

# Talk about why linearising keeps the squashed probability 
# distributed as a gaussian (yes, the probability itself is a random variable)

plt.ion()

def main():

    """
    Demostration Options
    """
    logging.basicConfig(level = logging.DEBUG)

    # If using parallel functionality, you must call this to set the appropriate
    # logging level
    gp.classifier.set_multiclass_logging_level(logging.DEBUG)

    np.random.seed(100)
    # Feature Generation Parameters and Demonstration Options
    SAVE_OUTPUTS = True # We don't want to make files everywhere for a demo.
    SHOW_RAW_BINARY = True
    test_range_min = -2.5
    test_range_max = +2.5
    test_ranges = (test_range_min, test_range_max)
    n_train = 500
    n_query = 1000
    n_dims  = 2   # <- Must be 2 for vis
    n_cores = None # number of cores for multi-class (None -> default: c-1)
    walltime = 300.0
    approxmethod = 'laplace' # 'laplace' or 'pls'
    multimethod = 'OVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
    responsename = 'probit' # 'probit' or 'logistic'
    batch_start = False
    entropy_threshold = None

    n_draws = 6
    n_draws_est = 2500
    rows_subplot = 2
    cols_subplot = 3

    assert rows_subplot * cols_subplot >= n_draws

    # Decision boundaries
    db1 = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & \
            ((x1 + x2) < 1.5)
    db2 = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) > 0.3)
    db3 = lambda x1, x2: ((x1 + x2) < 2) & ((x1 + x2) > -2.2)
    db4 = lambda x1, x2: ((x1 - 0.75)**2 + (x2 + 0.8)**2 > 0.3**2)
    db5 = lambda x1, x2: ((x1/2)**2 + x2**2 > 0.3)
    db6 = lambda x1, x2: (((x1)/8)**2 + (x2 + 1.5)**2 > 0.2**2)
    db7 = lambda x1, x2: (((x1)/8)**2 + ((x2 - 1.4)/1.25)**2 > 0.2**2)
    db4a = lambda x1, x2: ((x1 - 1.25)**2 + (x2 - 1.25)**2 > 0.5**2) & ((x1 - 0.75)**2 + (x2 + 1.2)**2 > 0.6**2) & ((x1 + 0.75)**2 + (x2 + 1.2)**2 > 0.3**2) & ((x1 + 1.3)**2 + (x2 - 1.3)**2 > 0.4**2)
    db5a = lambda x1, x2: ((x1/2)**2 + x2**2 > 0.3) & (x1 > 0)
    db5b = lambda x1, x2: ((x1/2)**2 + x2**2 > 0.3) & (x1 < 0) & ((x1 + 0.75)**2 + (x2 - 1.2)**2 > 0.6**2)
    db1a = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & \
            ((x1 + x2) < 1.6) | ((x1 + 0.75)**2 + (x2 + 1.2)**2 < 0.6**2)
    db1b = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & ((x1/2)**2 + (x2)**2 > 0.4**2) & \
            ((x1 + x2) < 1.5) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.4**2) | ((x1 + x2) > 2.1) & (x1 < 1.8) & (x2 < 1.8) # | (((x1 + 0.25)/4)**2 + (x2 + 1.5)**2 < 0.32**2) # & (((x1 + 0.25)/4)**2 + (x2 + 1.5)**2 > 0.18**2)
    db1c = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & ((x1/2)**2 + (x2)**2 > 0.4**2) & \
            ((x1 + x2) < 1.5) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.4**2) | ((x1 + x2) > 2.1) & (x1 < 1.8) & (x2 < 1.8) | (((x1 + 0.25)/4)**2 + (x2 + 1.75)**2 < 0.32**2) & (((x1 + 0.25)/4)**2 + (x2 + 1.75)**2 > 0.18**2)
    db8 = lambda x1, x2: (np.sin(2*x1 + 3*x2) > 0) | (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.4) & \
            ((x1 + x2) < 1.5) | (x1 < -1.9) | (x1 > +1.9) | (x2 < -1.9) | (x2 > +1.9) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.3**2)
    # db9 = lambda x1, x2: ((x1)**2 + (x2)**2 < 0.3**2) | ((x1)**2 + (x2)**2 > 0.5**2) |
    decision_boundary  = db1b # [db5b, db1c, db4a] # [db5b, db1c, db4a, db8, db6, db7]

    """
    Data Generation
    """

    # # # Training Points
    # shrink = 0.8
    # test_range_min *= shrink
    # test_range_max *= shrink
    # X1 = np.random.normal(loc = np.array([test_range_min, test_range_min]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # X2 = np.random.normal(loc = np.array([test_range_min, test_range_max]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # X3 = np.random.normal(loc = np.array([test_range_max, test_range_min]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # X4 = np.random.normal(loc = np.array([test_range_max, test_range_max]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # X5 = np.random.normal(loc = np.array([0, test_range_min]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # X6 = np.random.normal(loc = np.array([test_range_min, 0]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # X7 = np.random.normal(loc = np.array([test_range_max, 0]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # X8 = np.random.normal(loc = np.array([0, test_range_max]), scale = 0.9*np.ones(n_dims), size = (int(n_train/8), n_dims))
    # test_range_min /= shrink
    # test_range_max /= shrink

    # X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8), axis = 0)

    X = np.random.uniform(test_range_min + 0.5, test_range_max - 0.5, 
        size = (n_train, n_dims))
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    Xw, whitenparams = pre.whiten(X)

    # Query Points
    Xq = np.random.uniform(test_range_min, test_range_max, 
        size = (n_query, n_dims))
    xq1 = Xq[:, 0]
    xq2 = Xq[:, 1]

    Xqw = pre.whiten(Xq, whitenparams)

    n_train = X.shape[0]
    n_query = Xq.shape[0]
    logging.info('Training Points: %d' % n_train)
    # Training Labels
    y = gp.classifier.utils.make_decision(X, decision_boundary)
    y_unique = np.unique(y)
    assert y_unique.dtype == int

    if y_unique.shape[0] == 2:
        mycmap = cm.get_cmap(name = 'bone', lut = None)
        mycmap2 = cm.get_cmap(name = 'BrBG', lut = None)
    else:
        mycmap = cm.get_cmap(name = 'gist_rainbow', lut = None)
        mycmap2 = cm.get_cmap(name = 'gist_rainbow', lut = None)
    """
    Classifier Training
    """

    # Training
    fig = plt.figure()
    gp.classifier.utils.visualise_decision_boundary(
        test_range_min, test_range_max, decision_boundary)
    
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.title('Training Labels')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    plt.gca().patch.set_facecolor('gray')
    print('Plotted Training Set')

    plt.show()

    # Training
    print('===Begin Classifier Training===')
    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # User can choose to batch start each binary classifier with different
    # initial hyperparameters for faster training
    if batch_start:
        if y_unique.shape[0] == 2:
            initial_hyperparams = [100, 0.1, 0.1]
        elif multimethod == 'OVA':
            initial_hyperparams = [  [356.468, 0.762, 0.530], \
                                     [356.556, 0.836, 0.763], \
                                     [472.006, 1.648, 1.550], \
                                     [239.720, 1.307, 0.721] ]
        elif multimethod == 'AVA':
            initial_hyperparams = [ [14.9670, 0.547, 0.402],  \
                                    [251.979, 1.583, 1.318], \
                                    [420.376, 1.452, 0.750], \
                                    [780.641, 1.397, 1.682], \
                                    [490.353, 2.299, 1.526], \
                                    [73.999, 1.584, 0.954]]
        else:
            raise ValueError
        batch_config = gp.batch_start(optimiser_config, initial_hyperparams)
    else:
        batch_config = optimiser_config

    # Obtain the response function
    responsefunction = gp.classifier.responses.get(responsename)

    # Train the classifier!
    learned_classifier = gp.classifier.learn(Xw, y, kerneldef,
        responsefunction, batch_config, 
        multimethod = multimethod, approxmethod = approxmethod,
        train = True, ftol = 1e-6, processes = n_cores)

    # Print learned kernels
    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(print_function, 
                                            learned_classifier, y_unique)

    # Print the matrix of learned classifier hyperparameters
    logging.info('Matrix of learned hyperparameters')
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)
    
    """
    Classifier Prediction
    """
    # Prediction
    yq_prob = gp.classifier.predict(Xq, learned_classifier, 
        fusemethod = fusemethod)
    yq_pred = gp.classifier.classify(yq_prob, y)
    yq_entropy = gp.classifier.entropy(yq_prob)

    logging.info('Caching Predictor...')
    predictors = gp.classifier.query(learned_classifier, Xq)
    logging.info('Computing Expectance...')
    yq_exp_list = gp.classifier.expectance(learned_classifier, predictors)
    logging.info('Computing Covariance...')
    yq_cov_list = gp.classifier.covariance(learned_classifier, predictors)
    logging.info('Drawing from GP...')
    yq_draws = gp.classifier.draws(n_draws, yq_exp_list, yq_cov_list, 
        learned_classifier)
    logging.info('Computing Linearised Entropy...')
    yq_linearised_entropy = gp.classifier.linearised_entropy(
        yq_exp_list, yq_cov_list, learned_classifier)
    logging.info('Linearised Entropy is {0}'.format(yq_linearised_entropy))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        THE GAP BETWEEN ANALYSIS AND PLOTS
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""







    """
    Classifier Prediction Results (Plots)
    """

    logging.info('Plotting... please wait')

    Xq_plt = gp.classifier.utils.query_map(test_ranges, n_points = 250)
    Xqw_plt = pre.whiten(Xq_plt, whitenparams)
    yq_truth_plt = gp.classifier.utils.make_decision(Xq_plt, decision_boundary)

    fig = plt.figure(figsize = (15, 15 * 1.5))
    fontsize = 24
    axis_tick_font_size = 14

    """
    Plot: Ground Truth
    """

    # Training
    plt.subplot(3, 2, 1)
    gp.classifier.utils.visualise_map(yq_truth_plt, test_ranges, cmap = mycmap)
    plt.title('Ground Truth', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    gp.classifier.utils.visualise_decision_boundary(
        test_range_min, test_range_max, decision_boundary)
    logging.info('Plotted Prediction Labels')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    """
    Plot: Training Set
    """

    # Training
    plt.subplot(3, 2, 2)
    gp.classifier.utils.visualise_decision_boundary(
        test_range_min, test_range_max, decision_boundary)
    
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.title('Training Labels', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    plt.gca().patch.set_facecolor('gray')
    logging.info('Plotted Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """
    Plot: Query Computations
    """

    # Compute Linearised and True Entropy for plotting
    logging.info('Plot: Caching Predictor...')
    predictor_plt = gp.classifier.query(learned_classifier, Xqw_plt)
    logging.info('Plot: Computing Expectance...')
    expectance_latent_plt = \
        gp.classifier.expectance(learned_classifier, predictor_plt)
    logging.info('Plot: Computing Variance...')
    variance_latent_plt = \
        gp.classifier.variance(learned_classifier, predictor_plt)
    logging.info('Plot: Computing Linearised Entropy...')
    entropy_linearised_plt = gp.classifier.linearised_entropy(
        expectance_latent_plt, variance_latent_plt, learned_classifier)
    logging.info('Plot: Computing Equivalent Standard Deviation')
    eq_sd_plt = gp.classifier.equivalent_standard_deviation(
        entropy_linearised_plt)
    logging.info('Plot: Computing Prediction Probabilities...')
    yq_prob_plt = gp.classifier.predict_from_latent(
        expectance_latent_plt, variance_latent_plt, learned_classifier, 
        fusemethod = fusemethod)
    logging.info('Plot: Computing True Entropy...')
    yq_entropy_plt = gp.classifier.entropy(yq_prob_plt)
    logging.info('Plot: Computing Class Predicitons')
    yq_pred_plt = gp.classifier.classify(yq_prob_plt, y_unique)

    Xq_meas = gp.classifier.utils.query_map(test_ranges, n_points = 10)

    predictor_meas = gp.classifier.query(learned_classifier, Xq_meas)
    exp_meas = gp.classifier.expectance(learned_classifier, predictor_meas)
    cov_meas = gp.classifier.covariance(learned_classifier, predictor_meas)

    logging.info('Objective Measure: Computing Joint Linearised Entropy...')
    entropy_linearised_meas = gp.classifier.linearised_entropy(
        exp_meas, cov_meas, learned_classifier)
    logging.info('Objective Measure: Computing Monte Carlo Joint Entropy...')

    # start_time = time.clock()
    # entropy_monte_carlo_meas = gp.classifier.monte_carlo_joint_entropy(exp_meas, cov_meas, learned_classifier, n_draws = n_draws_est)
    # logging.info('Sampling took %.4f seconds' % (time.clock() - start_time))

    entropy_linearised_mean_meas = entropy_linearised_plt.mean()
    entropy_true_mean_meas = yq_entropy_plt.mean()

    mistake_ratio = (yq_truth_plt - yq_pred_plt).nonzero()[0].shape[0] / yq_truth_plt.shape[0]

    """
    Plot: Prediction Labels
    """

    # Query (Prediction Map)
    plt.subplot(3, 2, 3)
    gp.classifier.utils.visualise_map(yq_pred_plt, test_ranges, 
        boundaries = True, cmap = mycmap)
    plt.title('Prediction [Miss Ratio: %.3f %s]' % (100 * mistake_ratio, '%'), fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    logging.info('Plotted Prediction Labels')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """
    Plot: Prediction Entropy onto Training Set
    """

    # Query (Prediction Entropy)
    plt.subplot(3, 2, 4)
    gp.classifier.utils.visualise_map(yq_entropy_plt, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    plt.title('Prediction Information Entropy', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Prediction Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """
    Plot: Linearised Prediction Entropy onto Training Set
    """

    # Query (Linearised Entropy)
    plt.subplot(3, 2, 5)
    entropy_linearised_plt_min = entropy_linearised_plt.min()
    entropy_linearised_plt_max = entropy_linearised_plt.max()
    gp.classifier.utils.visualise_map(entropy_linearised_plt, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm, 
        vmin = -entropy_linearised_plt_max, vmax = entropy_linearised_plt_max)
    plt.title('Linearised Differential Entropy', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Linearised Prediction Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """
    Plot: Exponentiated Linearised Prediction Entropy onto Training Set
    """

    # Query (Linearised Entropy)
    plt.subplot(3, 2, 6)
    gp.classifier.utils.visualise_map(eq_sd_plt, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    plt.title('Equivalent Standard Deviation', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Exponentiated Linearised Prediction Entropy (Equivalent Standard Deviation) on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    plt.tight_layout()

    """
    Plot: Sample Query Predictions
    """  

    # Visualise Predictions
    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_decision_boundary(
        test_range_min, test_range_max, decision_boundary)
    plt.scatter(xq1, xq2, c = yq_pred, marker = 'x', cmap = mycmap)
    plt.title('Predicted Query Labels')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    plt.gca().patch.set_facecolor('gray')
    logging.info('Plotted Sample Query Labels')
    plt.gca().set_aspect('equal', adjustable = 'box')

    """
    Plot: Latent
    """  

    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(expectance_latent_plt, test_ranges, 
        cmap = cm.coolwarm)
    plt.title('Latent Expectance', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Latent Expectance on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(variance_latent_plt, test_ranges, 
        cmap = cm.coolwarm)
    plt.title('Latent Variance', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Latent Variance on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(yq_prob_plt, test_ranges, 
        cmap = cm.coolwarm)
    plt.title('Prediction Probabilities', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Prediction Probabilities on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    """
    Plot: Sample Query Draws
    """  

    # Visualise Predictions
    fig = plt.figure(figsize = (19.2, 10.8))
    for i in range(n_draws):
        plt.subplot(rows_subplot, cols_subplot, i + 1)
        gp.classifier.utils.visualise_decision_boundary(
            test_range_min, test_range_max, decision_boundary)
        plt.scatter(xq1, xq2, c = yq_draws[i], marker = 'x', cmap = mycmap)
        plt.title('Query Label Draws')
        plt.xlabel('x1')
        plt.ylabel('x2')
        cbar = plt.colorbar()
        cbar.set_ticks(y_unique)
        cbar.set_ticklabels(y_unique)
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))
        plt.gca().patch.set_facecolor('gray')
        logging.info('Plotted Sample Query Draws')

    """
    Save Outputs
    """  

    # Save all the figures
    if SAVE_OUTPUTS:
        save_directory = "binary_linearised_entropy/"
        full_directory = gp.classifier.utils.create_directories(
            save_directory, home_directory = '../Figures/', append_time = False)
        gp.classifier.utils.save_all_figures(full_directory)
        shutil.copy2('./binary_linearised_entropy.py', full_directory)

    logging.info('Modeling Done')


if __name__ == "__main__":
    main()

# DO TO: Put learned hyperparam in the title
# DO TO: Find joint entropy of the whole region and put it in the title
# TO DO: Find other measures of improvement (sum of entropy (linearised and true)) (sum of variances and standard deviation)