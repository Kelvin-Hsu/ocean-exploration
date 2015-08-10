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
import rh
import shelve

def main():

    """
    Demostration Options
    """

    # Set logging level
    logging.basicConfig(level = logging.DEBUG)
    gp.classifier.set_multiclass_logging_level(logging.DEBUG)

    np.random.seed(100)
    
    # Feature Generation Parameters and Demonstration Options
    SAVE_OUTPUTS = True
    test_range_min = -2.5
    test_range_max = +2.5
    test_ranges = (test_range_min, test_range_max)
    n_train = 50
    n_query = 1000
    n_dims  = 2   # <- Must be 2 for vis
    n_cores = 1 # number of cores for multi-class (None -> default: c-1)
    walltime = 300.0
    approxmethod = 'laplace' # 'laplace' or 'pls'
    multimethod = 'OVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
    responsename = 'probit' # 'probit' or 'logistic'
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
    db4a = lambda x1, x2: ((x1 - 1.25)**2 + (x2 - 1.25)**2 > 0.5**2) & ((x1 - 0.75)**2 + (x2 + 1.2)**2 > 0.6**2) & ((x1 + 0.75)**2 + (x2 + 1.2)**2 > 0.3**2) & ((x1 + 1.3)**2 + (x2 - 1.3)**2 > 0.4**2) & (x1 > -2) & ((x1 - x2) < 4.2)
    db5a = lambda x1, x2: ((x1/2)**2 + x2**2 > 0.3) & (x1 > 0)
    db5b = lambda x1, x2: ((x1/2)**2 + x2**2 > 0.25) & (x1 < 0.7) & ((x1 + 0.75)**2 + (x2 - 1.2)**2 > 0.6**2) & (x1 - x2 < 1.5)
    db1a = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & \
            ((x1 + x2) < 1.6) | ((x1 + 0.75)**2 + (x2 + 1.2)**2 < 0.6**2)
    db1b = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & ((x1/2)**2 + (x2)**2 > 0.4**2) & \
            ((x1 + x2) < 1.5) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.4**2) | ((x1 + x2) > 2.1) & (x1 < 1.8) & (x2 < 1.8) # | (((x1 + 0.25)/4)**2 + (x2 + 1.5)**2 < 0.32**2) # & (((x1 + 0.25)/4)**2 + (x2 + 1.5)**2 > 0.18**2)
    db1c = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & ((x1/2)**2 + (x2)**2 > 0.4**2) & \
            ((x1 + x2) < 1.5) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.4**2) | ((x1 + x2) > 2.1) & (x1 < 1.8) & (x2 < 1.8) | (((x1 + 0.25)/4)**2 + (x2 + 1.75)**2 < 0.32**2) & (((x1 + 0.25)/4)**2 + (x2 + 1.75)**2 > 0.18**2)
    db1d = lambda x1, x2: db1c(x1, x2) | (np.sin(4*(x1 + x2)) > 0)
    db8 = lambda x1, x2: (np.sin(2*x1 + 3*x2) > 0) | (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.4) & \
            ((x1 + x2) < 1.5) | (x1 < -1.9) | (x1 > +1.9) | (x2 < -1.9) | (x2 > +1.9) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.3**2)
    # db9 = lambda x1, x2: ((x1)**2 + (x2)**2 < 0.3**2) | ((x1)**2 + (x2)**2 > 0.5**2) |
    # db10 = lambda x1, x2: x1 - x2 > 1.5

    # circle = lambda x1, x2, c1, c2, r: ((x1 - c1)**2 + (x2 - c2)**2 < r**2)
    # ellipse = lambda x1, x2, c1, c2, a, b: (((x1 - c1)/a)**2 + ((x2 - c2)/b)**2 < 1)

    # def U(*args, **kwargs):
    #     if len(args) == 1:
    #         return np.random.uniform(0, args[0], **kwargs)
    #     else:
    #         return np.random.uniform(*args, **kwargs)

    # P = iter(np.random.uniform(test_range_min + 0.5, test_range_max - 0.5, size = 100))
    # R = iter(np.random.uniform(0.2, 0.8, size = 100))

    # c1 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c2 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c3 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c4 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c5 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c6 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c7 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c8 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c9 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))
    # c10 = lambda x1, x2: circle(x1, x2, next(P), next(P), next(R))

    # e1 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e2 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e3 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e4 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e5 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e6 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e7 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e8 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e9 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))
    # e10 = lambda x1, x2: ellipse(x1, x2, next(P), next(P), next(R), next(R))

    ellipse = lambda x1, x2, A: (((x1 - A[0])/A[2])**2 + ((x2 - A[1])/A[3])**2 < 1)

    n_ellipse = 40
    P = np.random.uniform(test_range_min, test_range_max, size = (n_ellipse, n_dims))
    B = np.random.uniform(0.1, 0.5, size = (n_ellipse, n_dims))
    A = np.concatenate((P, B), axis = 1)


    # dbce = lambda x1, x2: c1(x1, x2) | c2(x1, x2) | c3(x1, x2) | c4(x1, x2) | c5(x1, x2) | e6(x1, x2) | e7(x1, x2) | e8(x1, x2) | e9(x1, x2) | e10(x1, x2)
    dce = lambda x1, x2: np.array([ellipse(x1, x2, a) for a in A]).sum(axis = 0) > 0 

    db10 = lambda x1, x2: (np.sin(4*x1 - 4*x2) > 0) | (np.sin(4*x1 - 8*x2) > 0)
    decision_boundary  =  dce # [db5b, db1c, db4a] # [db5b, db1c, db4a, db8, db6, db7] # [db5b, db1c, db4a]

    """
    Data Generation
    """
    X = rh.utils.generate_tracks(1.8, 1.0, 10, 15, perturb_deg_scale = 5.0)
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
    logging.info('Plotted Training Set')

    plt.show()

    # Training
    logging.info('===Begin Classifier Training===')
    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # Obtain the response function
    responsefunction = gp.classifier.responses.get(responsename)

    # Train the classifier!
    learned_classifier = gp.classifier.learn(Xw, y, kerneldef,
        responsefunction, optimiser_config, 
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
    yq_prob = gp.classifier.predict(Xqw, learned_classifier, 
        fusemethod = fusemethod)
    yq_pred = gp.classifier.classify(yq_prob, y)
    yq_entropy = gp.classifier.entropy(yq_prob)

    logging.info('Caching Predictor...')
    predictors = gp.classifier.query(learned_classifier, Xqw)
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
    Xq_meas = gp.classifier.utils.query_map(test_ranges, n_points = 10)
    Xqw_plt = pre.whiten(Xq_plt, whitenparams)
    yq_truth_plt = gp.classifier.utils.make_decision(Xq_plt, decision_boundary)

    fig = plt.figure(figsize = (15 * 1.5, 15))
    fontsize = 24
    axis_tick_font_size = 14

    """
    Plot: Ground Truth
    """

    # Training
    plt.subplot(2, 3, 1)
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
    plt.subplot(2, 3, 2)
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

    Xqw_plt = pre.whiten(Xq_plt, whitenparams)
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

    Xqw_meas = pre.whiten(Xq_meas, whitenparams)
    predictor_meas = gp.classifier.query(learned_classifier, Xqw_meas)
    exp_meas = gp.classifier.expectance(learned_classifier, predictor_meas)
    cov_meas = gp.classifier.covariance(learned_classifier, predictor_meas)

    logging.info('Objective Measure: Computing Linearised Joint Entropy')
    start_time = time.clock()
    entropy_linearised_meas = gp.classifier.linearised_entropy(
        exp_meas, cov_meas, learned_classifier)
    logging.info('Computation took %.4f seconds' % (time.clock() - start_time))
    logging.info('Linearised Joint Entropy: %.4f' % entropy_linearised_meas)

    entropy_linearised_mean_meas = entropy_linearised_plt.mean()
    entropy_true_mean_meas = yq_entropy_plt.mean()

    mistake_ratio = (yq_truth_plt - yq_pred_plt).nonzero()[0].shape[0] / yq_truth_plt.shape[0]

    """
    Plot: Prediction Labels
    """

    # Query (Prediction Map)
    plt.subplot(2, 3, 3)
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
    plt.subplot(2, 3, 4)
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
    plt.subplot(2, 3, 5)
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
    plt.subplot(2, 3, 6)
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
        save_directory = "receding_horizon_informative_exploration/"
        full_directory = gp.classifier.utils.create_directories(
            save_directory, home_directory = '../Figures/', append_time = True)
        gp.classifier.utils.save_all_figures(full_directory)
        shutil.copy2('./receding_horizon_informative_exploration.py', full_directory)

    logging.info('Modeling Done')

    """
    Path Planning
    """
    METHOD = 'GREEDY'
    # np.random.seed(20)

    """ Setup Path Planning """
    xq_now = np.array([[0., 0.]])
    # xq_now = np.random.uniform(test_range_min, test_range_max, size = (1, n_dims))
    horizon = (test_range_max - test_range_min) + 0.5
    n_steps = 30

    if METHOD == 'GREEDY':
        horizon /= n_steps
        n_steps /= n_steps
        METHOD = 'MIE'

    theta_bound = np.deg2rad(180)
    theta_stack_init = np.deg2rad(10) * np.ones(n_steps)
    theta_stack_init[0] = np.deg2rad(180)
    theta_stack_low = -theta_bound * np.ones(n_steps)
    theta_stack_high = theta_bound * np.ones(n_steps)
    theta_stack_low[0] = 0.0
    theta_stack_high[0] = 2 * np.pi
    r = horizon/n_steps
    choice_walltime = 1500.0
    xtol_rel = 1e-2
    ftol_rel = 1e-4

    k_step = 1

    """ Initialise Values """

    # The observed data till now
    X_now = X.copy()
    y_now = y.copy()

    # Observe the current location
    yq_now = gp.classifier.utils.make_decision(xq_now[[-1]], 
        decision_boundary)

    # Add the observed data to the training set
    X_now = np.concatenate((X_now, xq_now[[-1]]), axis = 0)
    y_now = np.append(y_now, yq_now)

    # Add the new location to the array of travelled coordinates
    xq1_nows = xq_now[:, 0]
    xq2_nows = xq_now[:, 1]
    yq_nows = yq_now.copy()

    # Plot the current situation
    fig1 = plt.figure(figsize = (15, 15))
    fig2 = plt.figure(figsize = (15, 15))
    fig3 = plt.figure(figsize = (15, 15))
    fig4 = plt.figure(figsize = (15, 15))
    fig5 = plt.figure(figsize = (20, 20))

    # Start exploring
    i_trials = 0
    n_trials = 300
    entropy_linearised_array = np.nan * np.ones(n_trials)
    entropy_linearised_mean_array = np.nan * np.ones(n_trials)
    entropy_true_mean_array = np.nan * np.ones(n_trials)
    entropy_opt_array = np.nan * np.ones(n_trials)
    mistake_ratio_array = np.nan * np.ones(n_trials)

    m_step = 1
    while i_trials < n_trials:

        """ Path Planning """

        if m_step <= k_step:
            # Propose a place to observe
            xq_abs_opt, theta_stack_opt, entropy_opt = \
                rh.optimal_path(theta_stack_init, xq_now[-1], r, 
                    learned_classifier, whitenparams, test_ranges, 
                    theta_stack_low = theta_stack_low, theta_stack_high = theta_stack_high, 
                    walltime = choice_walltime, xtol_rel = xtol_rel, 
                    ftol_rel = ftol_rel, globalopt = False, objective = METHOD,
                    n_draws = n_draws_est)
            logging.info('Optimal Joint Entropy: %.5f' % entropy_opt)

            # m_step = rh.correct_lookahead_predictions(xq_abs_opt, learned_classifier, whitenparams, decision_boundary)
            logging.info('Taking %d steps' % m_step)
        else:
            m_step -= 1
            theta_stack_opt = theta_stack_init.copy()
            xq_abs_opt = rh.forward_path_model(theta_stack_init, r, xq_now[-1])
            logging.info('%d steps left' % m_step)

        xq_now = xq_abs_opt[:k_step]

        theta_stack_init = rh.shift_path(theta_stack_opt, 
            k_step = k_step)
        np.clip(theta_stack_init, theta_stack_low + 1e-4, theta_stack_high - 1e-4, 
            out = theta_stack_init)

        # Observe the current location
        yq_now = gp.classifier.utils.make_decision(xq_now, 
            decision_boundary)

        # Add the observed data to the training set
        X_now = np.concatenate((X_now, xq_now), axis = 0)
        y_now = np.append(y_now, yq_now)

        # Add the new location to the array of travelled coordinates
        xq1_nows = np.append(xq1_nows, xq_now[:, 0])
        xq2_nows = np.append(xq2_nows, xq_now[:, 1])
        yq_nows = np.append(yq_nows, yq_now)

        # Update that into the model
        Xw_now, whitenparams = pre.whiten(X_now)
        logging.info('Learning Classifier...')
        batch_config = \
            gp.classifier.batch_start(optimiser_config, learned_classifier)
        try:
            learned_classifier = gp.classifier.learn(Xw_now, y_now, kerneldef,
                responsefunction, batch_config, 
                multimethod = multimethod, approxmethod = approxmethod,
                train = True, ftol = 1e-6, processes = n_cores)
        except Exception as e:
            logging.warning('Training failed: {0}'.format(e))
            try:
                learned_classifier = gp.classifier.learn(Xw_now, y_now, kerneldef,
                    responsefunction, batch_config, 
                    multimethod = multimethod, approxmethod = approxmethod,
                    train = False, ftol = 1e-6, processes = n_cores)
            except Exception as e:
                logging.warning('Learning also failed: {0}'.format(e))
                pass    
        logging.info('Finished Learning')

        # This is the finite horizon optimal route
        xqw_abs_opt = pre.whiten(xq_abs_opt, whitenparams)
        xq1_proposed = xq_abs_opt[:, 0][k_step:]
        xq2_proposed = xq_abs_opt[:, 1][k_step:]
        yq_proposed = gp.classifier.classify(gp.classifier.predict(xqw_abs_opt, 
            learned_classifier), y_unique)[k_step:]

        """ Computing Analysis Maps """

        Xqw_plt = pre.whiten(Xq_plt, whitenparams)
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

        Xqw_meas = pre.whiten(Xq_meas, whitenparams)
        predictor_meas = gp.classifier.query(learned_classifier, Xqw_meas)
        exp_meas = gp.classifier.expectance(learned_classifier, predictor_meas)
        cov_meas = gp.classifier.covariance(learned_classifier, predictor_meas)

        logging.info('Objective Measure: Computing Linearised Joint Entropy')
        start_time = time.clock()
        entropy_linearised_meas = gp.classifier.linearised_entropy(
            exp_meas, cov_meas, learned_classifier)
        logging.info('Computation took %.4f seconds' % (time.clock() - start_time))
        logging.info('Linearised Joint Entropy: %.4f' % entropy_linearised_meas)

        entropy_linearised_mean_meas = entropy_linearised_plt.mean()
        entropy_true_mean_meas = yq_entropy_plt.mean()

        mistake_ratio = (yq_truth_plt - yq_pred_plt).nonzero()[0].shape[0] / yq_truth_plt.shape[0]

        entropy_linearised_array[i_trials] = entropy_linearised_meas
        entropy_linearised_mean_array[i_trials] = entropy_linearised_mean_meas
        entropy_true_mean_array[i_trials] = entropy_true_mean_meas
        entropy_opt_array[i_trials] = entropy_opt
        mistake_ratio_array[i_trials] = mistake_ratio
        

        # Find the bounds of the entropy predictions
        if entropy_linearised_plt.max() > 0:
            vmin1 = entropy_linearised_plt.min()
            vmax1 = entropy_linearised_plt.max()
        vmin2 = yq_entropy_plt.min()
        vmax2 = yq_entropy_plt.max()
        vmin3 = eq_sd_plt.min()
        vmax3 = eq_sd_plt.max()

        """ Linearised Entropy Map """

        # Prepare Figure 1
        plt.figure(fig1.number)
        plt.clf()
        plt.title('Linearised Differential Entropy', fontsize = fontsize)
        plt.xlabel('x1', fontsize = fontsize)
        plt.ylabel('x2', fontsize = fontsize)
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))

        # Plot linearised entropy
        gp.classifier.utils.visualise_map(entropy_linearised_plt, test_ranges, 
            cmap = cm.coolwarm, vmin = -vmax1, vmax = vmax1)
        plt.colorbar()

        # Plot training set on top
        plt.scatter(x1, x2, c = y, s = 40, marker = 'x', cmap = mycmap)

        # Plot the path on top
        plt.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none',
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_nows, xq2_nows, c = 'w')
        plt.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Plot the proposed path
        plt.scatter(xq1_proposed, xq2_proposed, c = yq_proposed, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_proposed, xq2_proposed, c = 'w')

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sentropy_linearised_step%d.png' 
            % (full_directory, i_trials + 1))

        """ Equivalent Standard Deviation Map """

        # Prepare Figure 2
        plt.figure(fig2.number)
        plt.clf()
        plt.title('Equivalent Standard Deviation', fontsize = fontsize)
        plt.xlabel('x1', fontsize = fontsize)
        plt.ylabel('x2', fontsize = fontsize)
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))

        # Plot linearised entropy
        gp.classifier.utils.visualise_map(eq_sd_plt, test_ranges, 
            cmap = cm.coolwarm, vmin = vmin3, vmax = vmax3)
        plt.colorbar()

        # Plot training set on top
        plt.scatter(x1, x2, c = y, s = 40, marker = 'x', cmap = mycmap)

        # Plot the path on top
        plt.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none',
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_nows, xq2_nows, c = 'w')
        plt.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Plot the proposed path
        plt.scatter(xq1_proposed, xq2_proposed, c = yq_proposed, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_proposed, xq2_proposed, c = 'w')

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%seq_sd_step%d.png' 
            % (full_directory, i_trials + 1))

        """ True Entropy Map """

        # Prepare Figure 3
        plt.figure(fig3.number)
        plt.clf()
        plt.title('Prediction Information Entropy', fontsize = fontsize)
        plt.xlabel('x1', fontsize = fontsize)
        plt.ylabel('x2', fontsize = fontsize)
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))

        # Plot true entropy
        gp.classifier.utils.visualise_map(yq_entropy_plt, test_ranges, 
            cmap = cm.coolwarm, vmin = vmin2, vmax = vmax2)
        plt.colorbar()

        # Plot training set on top
        plt.scatter(x1, x2, c = y, s = 40, marker = 'x', cmap = mycmap)

        # Plot the path on top
        plt.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none',
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_nows, xq2_nows, c = 'w')
        plt.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Plot the proposed path
        plt.scatter(xq1_proposed, xq2_proposed, c = yq_proposed, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_proposed, xq2_proposed, c = 'w')

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sentropy_true_step%d.png' 
            % (full_directory, i_trials + 1))

        """ Class Prediction Map """

        # Prepare Figure 4
        plt.figure(fig4.number)
        plt.clf()
        plt.title('Class predictions [Miss Ratio: %.3f %s]' % (100 * mistake_ratio, '%'), fontsize = fontsize)
        plt.xlabel('x1', fontsize = fontsize)
        plt.ylabel('x2', fontsize = fontsize)
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))

        # Plot class predictions
        gp.classifier.utils.visualise_map(yq_pred_plt, test_ranges, 
            boundaries = True, cmap = mycmap2, vmin = y_unique[0], vmax = y_unique[-1])
        cbar = plt.colorbar()
        cbar.set_ticks(y_unique)
        cbar.set_ticklabels(y_unique)


        # Plot training set on top
        plt.scatter(x1, x2, c = y, s = 40, marker = 'v', cmap = mycmap)

        # Plot the path on top
        plt.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, marker = 'o', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_nows, xq2_nows, c = 'w')
        plt.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Plot the proposed path
        plt.scatter(xq1_proposed, xq2_proposed, c = yq_proposed, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.plot(xq1_proposed, xq2_proposed, c = 'w')

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sclass_prediction_step%d.png' 
            % (full_directory, i_trials + 1))

        # Prepare Figure 5
        plt.figure(fig5.number)
        plt.clf()

        steps_array = np.arange(i_trials + 1) + 1
        ax = plt.subplot(5, 1, 1)
        plt.plot(steps_array, 100 * mistake_ratio_array[:(i_trials + 1)])
        plt.title('Percentage of Prediction Misses', fontsize = fontsize)
        plt.ylabel('Misses (%)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(5, 1, 2)
        plt.plot(steps_array, entropy_linearised_array[:(i_trials + 1)])
        plt.title('Joint Linearised Differential Entropy', fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(5, 1, 3)
        plt.plot(steps_array, entropy_linearised_mean_array[:(i_trials + 1)])
        plt.title('Average Marginalised Differential Entropy', fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(5, 1, 4)
        plt.plot(steps_array, entropy_true_mean_array[:(i_trials + 1)])
        plt.title('Average Marginalised Information Entropy', fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(5, 1, 5)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.plot(steps_array, entropy_opt_array[:(i_trials + 1)])
        plt.title('Entropy Metric of Proposed Path', fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)

        
        plt.xlabel('Steps', fontsize = fontsize)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_tick_font_size) 

        # Save the plot
        plt.tight_layout()
        plt.savefig('%shistory%d.png' 
            % (full_directory, i_trials + 1))
        logging.info('Plotted and Saved Iteration')

        # Move on to the next step
        i_trials += 1

    np.savez('%slearned_classifier.npz' % full_directory, 
        learned_classifier = learned_classifier)
    np.savez('%smistake_ratio_array.npz' % full_directory, 
        mistake_ratio_array = mistake_ratio_array)
    np.savez('%sentropy_linearised_array.npz' % full_directory, 
        entropy_linearised_array = entropy_linearised_array)
    np.savez('%sentropy_linearised_mean_array.npz' % full_directory, 
        entropy_linearised_mean_array = entropy_linearised_mean_array)
    np.savez('%sentropy_true_mean_array.npz' % full_directory, 
        entropy_true_mean_array = entropy_true_mean_array)
    np.savez('%sentropy_opt_array.npz' % full_directory, 
        entropy_opt_array = entropy_opt_array)

    np.savez('%shistory.npz' % full_directory, 
        learned_classifier = learned_classifier,
        mistake_ratio_array = mistake_ratio_array,
        entropy_linearised_array = entropy_linearised_array,
        entropy_linearised_mean_array = entropy_linearised_mean_array,
        entropy_true_mean_array = entropy_true_mean_array,
        entropy_opt_array = entropy_opt_array)

    # Shelf all work
    shelf = shelve.open('%sshelf.out' % full_directory, 'n') # 'n' for new
    for key in dir():
        try:
            shelf[key] = globals()[key]
        except TypeError:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()

    # Show everything!
    plt.show()

if __name__ == "__main__":
    main()

# DO TO: Put learned hyperparam in the title
# DO TO: Find joint entropy of the whole region and put it in the title
# TO DO: Find other measures of improvement (sum of entropy (linearised and true)) (sum of variances and standard deviation)