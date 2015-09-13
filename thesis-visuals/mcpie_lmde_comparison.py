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
import sea

# Talk about why linearising keeps the squashed probability 
# distributed as a gaussian (yes, the probability itself is a random variable)
def kerneldef(h, k):
    return h(1e-3, 1e5, 10) * k('gaussian', [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])

plt.ion()

def main():

    FONTSIZE = 50
    FONTNAME = 'Sans Serif'
    TICKSIZE = 24

    rcparams = {
        'backend': 'pdf',
        'axes.labelsize': TICKSIZE,
        'text.fontsize': FONTSIZE,
        'legend.fontsize': FONTSIZE,
        'xtick.labelsize': TICKSIZE,
        'ytick.labelsize': TICKSIZE,
        'text.usetex': True,
    }

    plt.rc_context(rcparams)

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
    decision_boundary  = [db5b, db1c, db4a] # db1b # [db5b, db1c, db4a] # [db5b, db1c, db4a, db8, db6, db7]

    """
    Data Generation
    """

    X = np.random.uniform(test_range_min + 0.5, test_range_max - 0.5, 
        size = (n_train, n_dims))
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    Xw, whitenparams = pre.whiten(X)

    n_train = X.shape[0]
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
    gp.classifier.utils.visualise_decision_boundary(plt.gca(),
        test_range_min, test_range_max, decision_boundary)
    
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.title('Training Labels')
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
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


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        THE GAP BETWEEN ANALYSIS AND PLOTS
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""







    """
    Classifier Prediction Results (Plots)
    """

    logging.info('Plotting... please wait')

    Xq = gp.classifier.utils.query_map(test_ranges, n_points = 250)
    Xqw = pre.whiten(Xq, whitenparams)
    yq_truth = gp.classifier.utils.make_decision(Xq, decision_boundary)

    fig = plt.figure(figsize = (19.2, 10.8))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    fontsize = 20
    axis_tick_font_size = 14

    """
    Plot: Ground Truth
    """

    # Training
    gp.classifier.utils.visualise_map(ax1, yq_truth, test_ranges, cmap = mycmap)
    ax1.set_title('Ground Truth', fontsize = fontsize)
    ax1.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax1.set_ylabel('$x_{2}$', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    gp.classifier.utils.visualise_decision_boundary(ax1,
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
    gp.classifier.utils.visualise_decision_boundary(ax2,
        test_range_min, test_range_max, decision_boundary)
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    ax2.set_title('Training Labels', fontsize = fontsize)
    ax2.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax2.set_ylabel('$x_{2}$', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    ax2.set_xlim((test_range_min, test_range_max))
    ax2.set_ylim((test_range_min, test_range_max))
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
    predictor = gp.classifier.query(learned_classifier, Xqw)

    logging.info('Plot: Computing Expectance...')
    exp = gp.classifier.expectance(learned_classifier, predictor)

    logging.info('Plot: Computing Variance...')
    var = gp.classifier.variance(learned_classifier, predictor)

    logging.info('Plot: Computing Linearised Entropy...')
    start_time = time.clock()
    yq_lmde = gp.classifier.linearised_model_differential_entropy(
        exp, var, learned_classifier)
    time_lmde = time.clock() - start_time

    logging.info('Plot: Computing Equivalent Standard Deviation')
    eqsd = gp.classifier.equivalent_standard_deviation(yq_lmde)

    logging.info('Plot: Computing Prediction Probabilities...')
    yq_prob = gp.classifier.predict_from_latent(exp, var, learned_classifier, 
        fusemethod = fusemethod)

    logging.info('Plot: Computing True Entropy...')
    start_time = time.clock()
    yq_pie = gp.classifier.entropy(yq_prob)
    time_pie = time.clock() - start_time

    n_draws_low = 25
    n_draws_med = 250
    n_draws_high = 2500

    logging.info('Plot: Computing MCPIE with %d samples' % n_draws_low)
    start_time = time.clock()
    yq_mcpie_low = gp.classifier.monte_carlo_prediction_information_entropy(exp, var, learned_classifier, n_draws = n_draws_low)
    time_mcpie_low = time.clock() - start_time

    logging.info('Plot: Computing MCPIE with %d samples' % n_draws_med)
    start_time = time.clock()
    yq_mcpie_med = gp.classifier.monte_carlo_prediction_information_entropy(exp, var, learned_classifier, n_draws = n_draws_med)
    time_mcpie_med = time.clock() - start_time

    logging.info('Plot: Computing MCPIE with %d samples' % n_draws_high)
    start_time = time.clock()
    yq_mcpie_high = gp.classifier.monte_carlo_prediction_information_entropy(exp, var, learned_classifier, n_draws = n_draws_high)
    time_mcpie_high = time.clock() - start_time

    logging.info('Plot: Computing Class Predicitons')
    yq_pred = gp.classifier.classify(yq_prob, y_unique)

    mistake_ratio = (yq_truth - yq_pred).nonzero()[0].shape[0] / yq_truth.shape[0]

    timing = {  'time_lmde': time_lmde,
                'time_pie': time_pie,
                'time_mcpie_low': time_mcpie_low,
                'time_mcpie_med': time_mcpie_med,
                'time_mcpie_high': time_mcpie_high}
    logging.info(timing)

    # """
    # THIS SECTION IS EXTRA FOR COLLECTING TIME COMPLEXITY DATA
    # PLEASE COMMENT OUT UNDER NORMAL CIRCUMSTANCES
    # """

    # # Compute Linearised and True Entropy for plotting
    # Xq = np.random.rand(1000, 2)
    # Xqw = pre.whiten(Xq, whitenparams)
    # logging.info('Plot: Caching Predictor...')
    # predictor = gp.classifier.query(learned_classifier, Xqw)

    # logging.info('Plot: Computing Expectance...')
    # exp = gp.classifier.expectance(learned_classifier, predictor)

    # logging.info('Plot: Computing Variance...')
    # cov = gp.classifier.covariance(learned_classifier, predictor)

    # logging.info('Plot: Computing Linearised Entropy...')
    # start_time = time.clock()
    # yq_lmde = gp.classifier.linearised_model_differential_entropy(
    #     exp, cov, learned_classifier)
    # time_lmde = time.clock() - start_time

    # logging.info('Plot: Computing Equivalent Standard Deviation')
    # eqsd = gp.classifier.equivalent_standard_deviation(yq_lmde)

    # logging.info('Plot: Computing Prediction Probabilities...')
    # yq_prob = gp.classifier.predict_from_latent(exp, cov, learned_classifier, 
    #     fusemethod = fusemethod)

    # logging.info('Plot: Computing True Entropy...')
    # start_time = time.clock()
    # yq_pie = gp.classifier.entropy(yq_prob)
    # time_pie = time.clock() - start_time

    # n_draws_low = 25
    # n_draws_med = 250
    # n_draws_high = 2500

    # logging.info('Plot: Computing MCPIE with %d samples' % n_draws_low)
    # start_time = time.clock()
    # yq_mcpie_low = gp.classifier.monte_carlo_prediction_information_entropy(exp, cov, learned_classifier, n_draws = n_draws_low)
    # time_mcpie_low = time.clock() - start_time

    # logging.info('Plot: Computing MCPIE with %d samples' % n_draws_med)
    # start_time = time.clock()
    # yq_mcpie_med = gp.classifier.monte_carlo_prediction_information_entropy(exp, cov, learned_classifier, n_draws = n_draws_med)
    # time_mcpie_med = time.clock() - start_time

    # logging.info('Plot: Computing MCPIE with %d samples' % n_draws_high)
    # start_time = time.clock()
    # yq_mcpie_high = gp.classifier.monte_carlo_prediction_information_entropy(exp, cov, learned_classifier, n_draws = n_draws_high)
    # time_mcpie_high = time.clock() - start_time

    # logging.info('Plot: Computing Class Predicitons')
    # yq_pred = gp.classifier.classify(yq_prob, y_unique)

    # timing = {  'time_lmde': time_lmde,
    #             'time_pie': time_pie,
    #             'time_mcpie_low': time_mcpie_low,
    #             'time_mcpie_med': time_mcpie_med,
    #             'time_mcpie_high': time_mcpie_high}
    # logging.info(timing)
    # print(yq_mcpie_high, yq_lmde)
    # return

    # """
    # Plot: Prediction Labels
    # """

    # Query (Prediction Map)
    gp.classifier.utils.visualise_map(ax3, yq_pred, test_ranges, 
        boundaries = True, cmap = mycmap)
    ax3.set_title('Prediction [Miss Ratio: %.1f %s]' % (100 * mistake_ratio, '\%'), fontsize = fontsize)
    ax3.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax3.set_ylabel('$x_{2}$', fontsize = fontsize)
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
    Plot: Prediction Information Entropy onto Training Set
    """

    gp.classifier.utils.visualise_map(ax4, yq_pie, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    ax4.set_title('Prediction Information Entropy', fontsize = fontsize)
    ax4.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax4.set_ylabel('$x_{2}$', fontsize = fontsize)
    plt.colorbar()
    ax4.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    ax4.set_xlim((test_range_min, test_range_max))
    ax4.set_ylim((test_range_min, test_range_max))
    logging.info('Plotted Prediction Information Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """
    Plot: Monte Carlo Prediction Information Entropy onto Training Set
    """

    gp.classifier.utils.visualise_map(ax5, yq_mcpie_high, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    ax5.set_title('M.C. Prediction Information Entropy', fontsize = fontsize, x = 0.45)
    ax5.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax5.set_ylabel('$x_{2}$', fontsize = fontsize)
    plt.colorbar()
    ax5.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    ax5.set_xlim((test_range_min, test_range_max))
    ax5.set_ylim((test_range_min, test_range_max))
    logging.info('Plotted Monte Carlo Prediction Information Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """
    Plot: Linearised Model Differential Entropy onto Training Set
    """

    yq_lmde_min = yq_lmde.min()
    yq_lmde_max = yq_lmde.max()
    gp.classifier.utils.visualise_map(ax6, yq_lmde, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm, 
        vmin = -yq_lmde_max, vmax = yq_lmde_max)
    ax6.set_title('L. Model Differential Entropy', fontsize = fontsize)
    ax6.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax6.set_ylabel('$x_{2}$', fontsize = fontsize)
    plt.colorbar()
    ax6.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    ax6.set_xlim((test_range_min, test_range_max))
    ax6.set_ylim((test_range_min, test_range_max))
    logging.info('Plotted Linearised Model Differential Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    # plt.show()
    fig.tight_layout()
    sea.vis.savefig(fig, './mcpie_lmde_comparison/mcpie_lmde_comparison.eps')

    fig = plt.figure(figsize = (19.2, 10.8/2))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    gp.classifier.utils.visualise_map(ax1, yq_mcpie_low, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    ax1.set_title('MCPIE with %d Samples' % n_draws_low, fontsize = fontsize)
    ax1.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax1.set_ylabel('$x_{2}$', fontsize = fontsize)
    plt.colorbar()
    ax1.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    ax1.set_xlim((test_range_min, test_range_max))
    ax1.set_ylim((test_range_min, test_range_max))
    logging.info('Plotted MCPIE with %d Samples on Training Set' % n_draws_low)
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    gp.classifier.utils.visualise_map(ax2, yq_mcpie_med, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    ax2.set_title('MCPIE with %d Samples' % n_draws_med, fontsize = fontsize)
    ax2.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax2.set_ylabel('$x_{2}$', fontsize = fontsize)
    plt.colorbar()
    ax2.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    ax2.set_xlim((test_range_min, test_range_max))
    ax2.set_ylim((test_range_min, test_range_max))
    logging.info('Plotted MCPIE with %d Samples on Training Set' % n_draws_med)
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    gp.classifier.utils.visualise_map(ax3, yq_mcpie_high, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    ax3.set_title('MCPIE with %d Samples' % n_draws_high, fontsize = fontsize)
    ax3.set_xlabel('$x_{1}$', fontsize = fontsize)
    ax3.set_ylabel('$x_{2}$', fontsize = fontsize)
    plt.colorbar()
    ax3.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    ax3.set_xlim((test_range_min, test_range_max))
    ax3.set_ylim((test_range_min, test_range_max))
    logging.info('Plotted MCPIE with %d Samples on Training Set' % n_draws_high)
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    # plt.show()
    fig.tight_layout()
    sea.vis.savefig(fig, './mcpie_lmde_comparison/mcpie_accuracy.eps')

    logging.info(timing)
    plt.show()

if __name__ == "__main__":
    main()

# DO TO: Put learned hyperparam in the title
# DO TO: Find joint entropy of the whole region and put it in the title
# TO DO: Find other measures of improvement (sum of entropy (linearised and true)) (sum of variances and standard deviation)