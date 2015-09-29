"""
Test 'generalClassifier'
Demonstration of general gp classifiers
Depending on the number of unique labels, this performs either binary 
classification or multiclass classification using binary classifiers 
(AVA or OVA)

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
import nlopt
import time
import sea

def main():

    """
    Command Argument Parser
    """
    # File options
    FOLDERNAME = ''.join(sys.argv[1:])
    FILENAME = sys.argv[0]

    # Trainint set generation options
    SEED = sea.io.parse('-seed', 50)
    E_MAX_SIZE = sea.io.parse('emax', 0.5)
    E_MIN_SIZE = sea.io.parse('emin', 0.25)
    N_ELLIPSE = sea.io.parse('-e', 20)
    RANGE = sea.io.parse('-range', 2.0)
    N_CLASS = sea.io.parse('-classes', 4)
    N_TRAIN = sea.io.parse('-ntrain', 300)

    # Modeling options
    WHITEFN = sea.io.parse('-whiten', 'pca')
    APPROX_METHOD = sea.io.parse('-approxmethod', 'laplace') # 'laplace' or 'pls'
    MULTI_METHOD = sea.io.parse('-multimethod', 'OVA')       # 'AVA' or 'OVA'
    FUSE_METHOD = sea.io.parse('-fusemethod', 'EXCLUSION')   # 'MODE' or 'EXCLUSION'
    RESPONSE_NAME = sea.io.parse('-response', 'probit')      # 'probit' or 'logistic'

    # Display options
    SHOW_TRAIN = sea.io.parse('-showtrain', False)
    SAVE_OUTPUTS = sea.io.parse('-save', False)

    """
    Analysis Options
    """
    # Set logging level
    logging.basicConfig(level = logging.DEBUG)
    gp.classifier.set_multiclass_logging_level(logging.DEBUG)
    # gp.classifier.partools.set_log_level(logging.DEBUG)

    # Feature Generation Parameters and Demonstration Options
    RANGES = (-RANGE, +RANGE)
    N_DIMS  = 2     # <- Must be 2 for vis
    N_CORES = None  # number of cores for learning (None -> default: c-1)
    N_QUERY = 250
    WALLTIME = 300.0
    
    # Generate the decision boundary with ellipses
    np.random.seed(SEED)
    decision_boundary = \
        gp.classifier.utils.generate_elliptical_decision_boundaries(RANGES, 
        min_size = E_MIN_SIZE, max_size = E_MAX_SIZE, 
        n_class = N_CLASS, n_ellipse = N_ELLIPSE, n_dims = N_DIMS)

    # Choose the whitening function
    if (WHITEFN == 'none') or (WHITEFN == 'NONE'):
        whitefn = sea.feature.black_fn
    elif (WHITEFN == 'pca') or (WHITEFN == 'PCA'):
        whitefn = pre.whiten
    elif (WHITEFN == 'standardise') or (WHITEFN == 'STANDARDISE'):
        whitefn = pre.standardise

    """
    Data Generation
    """
    # Generate training data and whiten it
    X = np.random.uniform(*RANGES, size = (N_TRAIN, N_DIMS))
    x1 = X[:, 0]
    x2 = X[:, 1]
    Xw, whitenparams = whitefn(X)

    # Create query data in grid form for visualisation
    Xq_plt = gp.classifier.utils.query_map(RANGES, n_points = N_QUERY)

    # Actual number of training points (in case it changes)
    n_train = X.shape[0]
    logging.info('Training Points: %d' % n_train)

    # Obtain training labels
    y = gp.classifier.utils.make_decision(X, decision_boundary)
    y_unique = np.unique(y)
    assert y_unique.shape[0] == N_CLASS

    """
    Plot Options
    """
    # Display fonts
    fontsize = 24
    axis_tick_font_size = 14
    
    # Choose an appropriate colormap
    if y_unique.shape[0] == 2:
        mycmap = cm.get_cmap(name = 'bone', lut = None)
    else:
        mycmap = cm.get_cmap(name = 'gist_rainbow', lut = None)

    """
    Classifier Training
    """
    if SHOW_TRAIN:
        fig = plt.figure()
        gp.classifier.utils.visualise_decision_boundary(plt.gca(),
            RANGES[0], RANGES[1], decision_boundary)
        plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
        plt.title('Training Labels')
        plt.xlabel('x1')
        plt.ylabel('x2')
        cbar = plt.colorbar()
        cbar.set_ticks(y_unique)
        cbar.set_ticklabels(y_unique)
        plt.xlim(RANGES)
        plt.ylim(RANGES)
        plt.gca().patch.set_facecolor('gray')
        logging.info('Plotted Training Set')
        plt.tight_layout()
        plt.show()

    kerneldef = sea.model.kerneldef2
    
    # Set optimiser configuration and obtain the response function
    logging.info('===Begin Classifier Training===')
    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = WALLTIME
    responsefunction = gp.classifier.responses.get(RESPONSE_NAME)

    # Train the classifier!
    learned_classifier = gp.classifier.learn(Xw, y, kerneldef,
        responsefunction, optimiser_config, 
        multimethod = MULTI_METHOD, approxmethod = APPROX_METHOD,
        train = True, ftol = 1e-6, processes = N_CORES)

    # Print learned kernels
    logging.info('Learned kernels:')
    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(
        print_function, learned_classifier, y_unique)

    # Print the matrix of learned classifier hyperparameters
    logging.info('Matrix of learned hyperparameters')
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)


    """
    Classifier Prediction Results (Plots)
    """

    """Feature Generation and Query Computations"""
    Xqw_plt = whitefn(Xq_plt, whitenparams)
    yq_truth_plt = gp.classifier.utils.make_decision(Xq_plt, decision_boundary)

    logging.info('Plot: Caching Predictor...')
    predictor_plt = gp.classifier.query(learned_classifier, Xqw_plt)
    logging.info('Plot: Computing Expectance...')
    exp_plt = gp.classifier.expectance(learned_classifier, predictor_plt)
    logging.info('Plot: Computing Variance...')
    var_plt = gp.classifier.variance(learned_classifier, predictor_plt)
    logging.info('Plot: Computing Linearised Model Differential Entropy...')
    yq_lmde_plt = gp.classifier.linearised_model_differential_entropy(
        exp_plt, var_plt, learned_classifier)
    logging.info('Plot: Computing Equivalent Standard Deviation...')
    yq_sd_plt = gp.classifier.equivalent_standard_deviation(yq_lmde_plt)
    logging.info('Plot: Computing Expected Predictive Probabilities...')
    yq_prob_plt = gp.classifier.predict_from_latent(exp_plt, var_plt, 
        learned_classifier, fusemethod = FUSE_METHOD)
    logging.info('Plot: Computing Prediction Information Entropy...')
    yq_entropy_plt = gp.classifier.entropy(yq_prob_plt)
    logging.info('Plot: Computing Class Predictions...')
    yq_pred_plt = gp.classifier.classify(yq_prob_plt, y_unique)

    mistake_ratio = sea.model.miss_ratio(yq_pred_plt, yq_truth_plt)

    """Plot: Ground Truth"""
    fig = plt.figure(figsize = (15 * 1.5, 15))

    plt.subplot(2, 3, 1)
    gp.classifier.utils.visualise_map(plt.gca(), yq_truth_plt, RANGES, 
        cmap = mycmap)
    plt.title('Ground Truth', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    gp.classifier.utils.visualise_decision_boundary(plt.gca(),
        RANGES[0], RANGES[1], decision_boundary)
    logging.info('Plotted Prediction Labels')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    """Plot: Training Set"""
    plt.subplot(2, 3, 2)
    gp.classifier.utils.visualise_decision_boundary(plt.gca(),
        RANGES[0], RANGES[1], decision_boundary)
    
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.title('Training Labels', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    plt.xlim(RANGES)
    plt.ylim(RANGES)
    plt.gca().patch.set_facecolor('gray')
    logging.info('Plotted Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    """Plot: Prediction Labels"""
    plt.subplot(2, 3, 3)
    gp.classifier.utils.visualise_map(plt.gca(), yq_pred_plt, RANGES, 
        boundaries = True, cmap = mycmap, 
        vmin = y_unique[0], vmax = y_unique[-1])
    plt.title('Prediction [Miss Ratio: %.3f %s]' % (100 * mistake_ratio, '%'), 
        fontsize = fontsize)
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
        
    """Plot: Prediction Information Entropy onto Training Set"""
    plt.subplot(2, 3, 4)
    gp.classifier.utils.visualise_map(plt.gca(), yq_entropy_plt, RANGES, 
        cmap = cm.coolwarm)
    plt.title('Prediction Information Entropy', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim(RANGES)
    plt.ylim(RANGES)
    logging.info('Plotted Prediction Information Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """Plot: Linearised Model Differential Entropy onto Training Set"""
    plt.subplot(2, 3, 5)
    yq_lmde_plt_min = yq_lmde_plt.min()
    yq_lmde_plt_max = yq_lmde_plt.max()
    gp.classifier.utils.visualise_map(plt.gca(), yq_lmde_plt, RANGES, 
        cmap = cm.coolwarm, 
        vmin = -yq_lmde_plt_max, vmax = yq_lmde_plt_max)
    plt.title('L. Model Differential Entropy', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim(RANGES)
    plt.ylim(RANGES)
    logging.info('Plotted Linearised Model Differential Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """Plot: Equivalent Standard Deviation onto Training Set"""
    plt.subplot(2, 3, 6)
    gp.classifier.utils.visualise_map(plt.gca(), yq_sd_plt, RANGES, 
        cmap = cm.coolwarm)
    plt.title('Equivalent Standard Deviation', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim(RANGES)
    plt.ylim(RANGES)
    logging.info('Plotted Equivalent Standard Deviation on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    plt.tight_layout()

    """
    Save Outputs
    """  
    if SAVE_OUTPUTS:
        save_directory = "%s/" % FOLDERNAME
        full_directory = gp.classifier.utils.create_directories(
            save_directory, home_directory = './Figures/', append_time = True,
            casual_format = True)
        gp.classifier.utils.save_all_figures(full_directory)

    logging.info('Modeling Done')

    # Show everything!
    plt.show()

if __name__ == "__main__":
    main()