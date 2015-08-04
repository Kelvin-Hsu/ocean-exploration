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

    # np.random.seed(50)
    # Feature Generation Parameters and Demonstration Options
    SAVE_OUTPUTS = True # We don't want to make files everywhere for a demo.
    SHOW_RAW_BINARY = True
    test_range_min = -2.0
    test_range_max = +2.0
    test_ranges = (test_range_min, test_range_max)
    n_train = 100
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
            ((x1 + x2) < 1.5) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.4**2) | ((x1 + x2) > 2.25) & (x1 < 1.75) & (x2 < 1.75) # | (((x1 + 0.25)/4)**2 + (x2 + 1.5)**2 < 0.32**2) # & (((x1 + 0.25)/4)**2 + (x2 + 1.5)**2 > 0.18**2)
    db1c = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & ((x1/2)**2 + (x2)**2 > 0.4**2) & \
            ((x1 + x2) < 1.5) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.4**2) | ((x1 + x2) > 2.25) & (x1 < 1.75) & (x2 < 1.75) | (((x1 + 0.25)/4)**2 + (x2 + 1.75)**2 < 0.32**2) & (((x1 + 0.25)/4)**2 + (x2 + 1.75)**2 > 0.18**2)
    db8 = lambda x1, x2: (np.sin(2*x1 + 3*x2) > 0) | (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.4) & \
            ((x1 + x2) < 1.5) | (x1 < -1.9) | (x1 > +1.9) | (x2 < -1.9) | (x2 > +1.9) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.3**2)
    # db9 = lambda x1, x2: ((x1)**2 + (x2)**2 < 0.3**2) | ((x1)**2 + (x2)**2 > 0.5**2) |
    decision_boundary  = [db5b, db1c, db4a] # [db5b, db1c, db4a, db8, db6, db7]

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

    X = np.random.uniform(test_range_min, test_range_max, 
        size = (n_train, n_dims))

    # X_s = np.array([[0.0, 0.0], [-0.2, 0.3], [-0.1, -0.1], [0.05, 0.25], [-1.1, 0.0], [-0.5, 0.0], [-0.4, -0.7], [-0.1, -0.1], [test_range_min, test_range_min], [test_range_min, test_range_max], [test_range_max, test_range_max], [test_range_max, test_range_min]])
    # X_f = np.array([[1.4, 1.6], [1.8, 1.2], [-1.24, 1.72], [-1.56, -1.9], [-1.9, 1.0], [-0.5, -1.2], [-1.4, -1.9], [0.4, -1.2], [test_range_min, test_range_max], [test_range_max, test_range_max], [test_range_max, test_range_min], [test_range_min, test_range_min]])
    # n_track = 30
    # X_s = np.random.uniform(test_range_min, test_range_max, size = (n_track, n_dims))
    # X_f = test_range_max * np.random.standard_cauchy(n_track * n_dims).reshape(n_track, n_dims)
    # X_f = np.random.uniform(test_range_min, test_range_max, size = (n_track, n_dims))
    # X = generate_line_paths(X_s, X_f, n_points = 15)
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    # Query Points
    Xq = np.random.uniform(test_range_min, test_range_max, 
        size = (n_query, n_dims))
    xq1 = Xq[:, 0]
    xq2 = Xq[:, 1]

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
    learned_classifier = gp.classifier.learn(X, y, kerneldef,
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
    yq_truth_plt = gp.classifier.utils.make_decision(Xq_plt, decision_boundary)

    """
    Plot: Ground Truth
    """

    # Training
    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(yq_truth_plt, test_ranges, cmap = mycmap)
    plt.title('Ground Truth')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    gp.classifier.utils.visualise_decision_boundary(
        test_range_min, test_range_max, decision_boundary)
    logging.info('Plotted Prediction Labels')

    """
    Plot: Training Set
    """

    # Training
    fig = plt.figure(figsize = (15, 15))
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

    """
    Plot: Query Computations
    """

    # Compute Linearised and True Entropy for plotting
    logging.info('Plot: Caching Predictor...')
    predictor_plt = gp.classifier.query(learned_classifier, Xq_plt)
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
        entropy_linearised_plt, y_unique.shape[0])
    logging.info('Plot: Computing Prediction Probabilities...')
    yq_prob_plt = gp.classifier.predict_from_latent(
        expectance_latent_plt, variance_latent_plt, learned_classifier, 
        fusemethod = fusemethod)
    logging.info('Plot: Computing True Entropy...')
    yq_entropy_plt = gp.classifier.entropy(yq_prob_plt)
    logging.info('Plot: Computing Class Predicitons')
    yq_pred_plt = gp.classifier.classify(yq_prob_plt, y_unique)

    if isinstance(learned_classifier, list):
        logging.info('Plot: Computing Naive Linearised Entropy...')
        args = [(expectance_latent_plt[i], variance_latent_plt[i], 
            learned_classifier[i]) for i in range(len(learned_classifier))]
        entropy_linearised_naive_plt = \
            np.array(parmap.starmap(gp.classifier.linearised_entropy, 
                args)).sum(axis = 0)


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

    if isinstance(learned_classifier, list) & False:

        """
        Plot: Latent Function Expectance
        """

        for i in range(len(expectance_latent_plt)):
            fig = plt.figure(figsize = (15, 15))
            gp.classifier.utils.visualise_map(
                expectance_latent_plt[i], test_ranges, 
                levels = [0.0], 
                vmin = -np.max(np.abs(expectance_latent_plt[i])), 
                vmax = np.max(np.abs(expectance_latent_plt[i])), 
                cmap = cm.coolwarm)
            plt.title('Latent Funtion Expectance %s' 
                % gp.classifier.utils.binary_classifier_name(
                    learned_classifier[i], y_unique))
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
            plt.xlim((test_range_min, test_range_max))
            plt.ylim((test_range_min, test_range_max))
            logging.info('Plotted Latent Function Expectance on Training Set')

        """
        Plot: Latent Function Variance
        """

        for i in range(len(variance_latent_plt)):
            fig = plt.figure(figsize = (15, 15))
            gp.classifier.utils.visualise_map(
                variance_latent_plt[i], test_ranges, 
                cmap = cm.coolwarm)
            plt.title('Latent Funtion Variance %s' 
                % gp.classifier.utils.binary_classifier_name(
                    learned_classifier[i], y_unique))
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
            plt.xlim((test_range_min, test_range_max))
            plt.ylim((test_range_min, test_range_max))
            logging.info('Plotted Latent Function Variance on Training Set')

        """
        Plot: Prediction Probabilities
        """

        for i in range(len(yq_prob_plt)):
            fig = plt.figure(figsize = (15, 15))
            gp.classifier.utils.visualise_map(yq_prob_plt[i], test_ranges, 
                levels = [0.5], cmap = cm.coolwarm)
            plt.title('Prediction Probabilities (Class %d)' % y_unique[i])
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
            plt.xlim((test_range_min, test_range_max))
            plt.ylim((test_range_min, test_range_max))
            logging.info('Plotted Prediction Probabilities on Training Set')

    """
    Plot: Prediction Labels
    """

    # Query (Prediction Map)
    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(yq_pred_plt, test_ranges, 
        boundaries = True, cmap = mycmap)
    plt.title('Prediction [Miss Ratio: %.3f %s]' % (100 * mistake_ratio, '%'))
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    logging.info('Plotted Prediction Labels')

    """
    Plot: Prediction Entropy onto Training Set
    """

    # Query (Prediction Entropy)
    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(yq_entropy_plt, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    plt.title('Prediction Entropy [ACE = %.4f]' 
            % (entropy_true_mean_meas))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Prediction Entropy on Training Set')

    """
    Plot: Linearised Prediction Entropy onto Training Set
    """

    # Query (Linearised Entropy)
    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(entropy_linearised_plt, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    plt.title('Linearised Prediction Entropy [FLE = %.4f, ALE = %.4f]' 
            % (entropy_linearised_meas, entropy_linearised_mean_meas))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Linearised Prediction Entropy on Training Set')

    """
    Plot: Exponentiated Linearised Prediction Entropy onto Training Set
    """

    # Query (Linearised Entropy)
    fig = plt.figure(figsize = (15, 15))
    gp.classifier.utils.visualise_map(eq_sd_plt, test_ranges, 
        threshold = entropy_threshold, cmap = cm.coolwarm)
    plt.title('Equivalent standard deviation')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    logging.info('Plotted Exponentiated Linearised Prediction Entropy (Equivalent Standard Deviation) on Training Set')

    """
    Plot: Naive Linearised Prediction Entropy onto Training Set
    """

    if isinstance(learned_classifier, list):
        # Query (Naive Linearised Entropy)
        fig = plt.figure(figsize = (15, 15))
        gp.classifier.utils.visualise_map(
            entropy_linearised_naive_plt, test_ranges, 
            threshold = entropy_threshold, cmap = cm.coolwarm)
        plt.title('Naive Linearised Prediction Entropy')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
        plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))
        logging.info('Plotted Naive Linearised Prediction Entropy on Training Set')

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
        save_directory = "response_%s_approxmethod_%s" \
        "_training_%d_query_%d_walltime_%d" \
        "_method_%s_fusemethod_%s/" \
            % ( responsename, approxmethod, 
                n_train, n_query, walltime, 
                multimethod, fusemethod)
        full_directory = gp.classifier.utils.create_directories(
            save_directory, home_directory = 'Figures/', append_time = True)
        gp.classifier.utils.save_all_figures(full_directory)
        shutil.copy2('./receding_horizon_path_planning.py', full_directory)

    logging.info('Modeling Done')

    """
    Path Planning
    """

    """ Setup Path Planning """
    xq_now = np.array([[0., 0.]])
    horizon = (test_range_max - test_range_min) + 0.5
    n_steps = 30

    theta_bound = np.deg2rad(30)
    theta_add_init = -np.deg2rad(10) * np.ones(n_steps)
    theta_add_init[0] = np.deg2rad(180)
    theta_add_low = -theta_bound * np.ones(n_steps)
    theta_add_high = theta_bound * np.ones(n_steps)
    theta_add_low[0] = 0.0
    theta_add_high[0] = 2 * np.pi
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
    n_trials = 2000
    entropy_linearised_array = np.nan * np.ones(n_trials)
    # entropy_monte_carlo_array = np.nan * np.ones(n_trials)
    entropy_linearised_mean_array = np.nan * np.ones(n_trials)
    entropy_true_mean_array = np.nan * np.ones(n_trials)
    entropy_opt_array = np.nan * np.ones(n_trials)
    mistake_ratio_array = np.nan * np.ones(n_trials)
    m_step = 0
    while i_trials < n_trials:

        """ Path Planning """

        print(m_step)
        print(k_step)
        if m_step <= k_step:
            # Propose a place to observe
            xq_abs_opt, theta_add_opt, entropy_opt = \
                go_optimised_path(theta_add_init, xq_now[-1], r, 
                    learned_classifier, test_ranges,
                    theta_add_low = theta_add_low, theta_add_high = theta_add_high, 
                    walltime = choice_walltime, xtol_rel = xtol_rel, 
                    ftol_rel = ftol_rel, globalopt = False, objective = 'LE',
                    n_draws = n_draws_est)
            logging.info('Optimal Joint Entropy: %.5f' % entropy_opt)

            m_step = keep_going_until_surprise(xq_abs_opt, learned_classifier, 
                decision_boundary)
            logging.info('Taking %d steps' % m_step)
        else:
            m_step -= 1
            theta_add_opt = theta_add_init.copy()
            xq_abs_opt = forward_path_model(theta_add_init, r, xq_now[-1])
            logging.info('%d steps left' % m_step)

        xq_now = xq_abs_opt[:k_step]

        theta_add_init = initiate_with_continuity(theta_add_opt, 
            k_step = k_step)
        np.clip(theta_add_init, theta_add_low + 1e-4, theta_add_high - 1e-4, 
            out = theta_add_init)

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
        logging.info('Learning Classifier...')
        batch_config = \
            gp.classifier.batch_start(optimiser_config, learned_classifier)
        try:
            learned_classifier = gp.classifier.learn(X_now, y_now, kerneldef,
                responsefunction, batch_config, 
                multimethod = multimethod, approxmethod = approxmethod,
                train = True, ftol = 1e-6, processes = n_cores)
        except Exception as e:
            logging.warning(e)
            try:
                learned_classifier = gp.classifier.learn(X_now, y_now, kerneldef,
                    responsefunction, batch_config, 
                    multimethod = multimethod, approxmethod = approxmethod,
                    train = False, ftol = 1e-6, processes = n_cores)
            except Exception as e:
                logging.warning(e)
                pass    
        logging.info('Finished Learning')

        # This is the finite horizon optimal route
        xq1_proposed = xq_abs_opt[:, 0][k_step:]
        xq2_proposed = xq_abs_opt[:, 1][k_step:]
        yq_proposed = gp.classifier.classify(gp.classifier.predict(xq_abs_opt, 
            learned_classifier), y_unique)[k_step:]

        """ Computing Analysis Maps """

        # Compute Linearised and True Entropy for plotting
        logging.info('Plot: Caching Predictor...')
        predictor_plt = gp.classifier.query(learned_classifier, Xq_plt)
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
            entropy_linearised_plt, y_unique.shape[0])
        logging.info('Plot: Computing Prediction Probabilities...')
        yq_prob_plt = gp.classifier.predict_from_latent(
            expectance_latent_plt, variance_latent_plt, learned_classifier, 
            fusemethod = fusemethod)
        logging.info('Plot: Computing True Entropy...')
        yq_entropy_plt = gp.classifier.entropy(yq_prob_plt)
        logging.info('Plot: Computing Class Predicitons')
        yq_pred_plt = gp.classifier.classify(yq_prob_plt, y_unique)

        
        predictor_meas = gp.classifier.query(learned_classifier, Xq_meas)
        exp_meas = gp.classifier.expectance(learned_classifier, predictor_meas)
        cov_meas = gp.classifier.covariance(learned_classifier, predictor_meas)

        logging.info('Objective Measure: Computing Linearised Joint Entropy')
        start_time = time.clock()
        entropy_linearised_meas = gp.classifier.linearised_entropy(
            exp_meas, cov_meas, learned_classifier)
        logging.info('Computation took %.4f seconds' % (time.clock() - start_time))
        logging.info('Linearised Joint Entropy: %.4f' % entropy_linearised_meas)
        # logging.info('Objective Measure: Computing Monte Carlo Joint Entropy...')
        # start_time = time.clock()
        # entropy_monte_carlo_meas = gp.classifier.monte_carlo_joint_entropy(exp_meas, cov_meas, learned_classifier, n_draws = n_draws_est)
        # logging.info('Computation took %.4f seconds' % (time.clock() - start_time))
        # logging.info('Monte Carlo Joint Entropy: %.4f' % entropy_monte_carlo_meas)

        entropy_linearised_mean_meas = entropy_linearised_plt.mean()
        entropy_true_mean_meas = yq_entropy_plt.mean()

        mistake_ratio = (yq_truth_plt - yq_pred_plt).nonzero()[0].shape[0] / yq_truth_plt.shape[0]

        entropy_linearised_array[i_trials] = entropy_linearised_meas
        # entropy_monte_carlo_array[i_trials] = entropy_monte_carlo_meas
        entropy_linearised_mean_array[i_trials] = entropy_linearised_mean_meas
        entropy_true_mean_array[i_trials] = entropy_true_mean_meas
        entropy_opt_array[i_trials] = entropy_opt
        mistake_ratio_array[i_trials] = mistake_ratio
        

        # Find the bounds of the entropy predictions
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
        plt.title('Linearised entropy [horizon = %.2f, FLE = %.2f, ALE = %.2f, ACE = %.2f, TPE = %.2f]' 
            % (horizon, entropy_linearised_meas, entropy_linearised_mean_meas, entropy_true_mean_meas, entropy_opt))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))

        # Plot linearised entropy
        gp.classifier.utils.visualise_map(entropy_linearised_plt, test_ranges, 
            cmap = cm.coolwarm, vmin = vmin1, vmax = vmax1)
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

        # Save the plot
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sentropy_linearised_step%d.png' 
            % (full_directory, i_trials + 1))

        """ Equivalent Standard Deviation Map """

        # Prepare Figure 2
        plt.figure(fig2.number)
        plt.clf()
        plt.title('Equivalent SD [horizon = %.2f, FLE = %.2f, ALE = %.2f, ACE = %.2f, TPE = %.2f]' 
            % (horizon, entropy_linearised_meas, entropy_linearised_mean_meas, entropy_true_mean_meas, entropy_opt))
        plt.xlabel('x1')
        plt.ylabel('x2')
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

        # Save the plot
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%seq_sd_step%d.png' 
            % (full_directory, i_trials + 1))

        """ True Entropy Map """

        # Prepare Figure 3
        plt.figure(fig3.number)
        plt.clf()
        plt.title('True entropy [horizon = %.2f, FLE = %.2f, ALE = %.2f, ACE = %.2f, TPE = %.2f]' 
            % (horizon, entropy_linearised_meas, entropy_linearised_mean_meas, entropy_true_mean_meas, entropy_opt))
        plt.xlabel('x1')
        plt.ylabel('x2')
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

        # Save the plot
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sentropy_true_step%d.png' 
            % (full_directory, i_trials + 1))

        """ Class Prediction Map """

        # Prepare Figure 4
        plt.figure(fig4.number)
        plt.clf()
        plt.title('Class predictions [Miss Ratio: %.3f %s]' % (100 * mistake_ratio, '%'))
        plt.xlabel('x1')
        plt.ylabel('x2')
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

        # Save the plot
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sclass_prediction_step%d.png' 
            % (full_directory, i_trials + 1))

        # Prepare Figure 5
        plt.figure(fig5.number)
        plt.clf()

        plt.subplot(5, 1, 1)
        plt.plot(np.arange(i_trials + 1), entropy_linearised_array[:(i_trials + 1)])
        plt.title('Field Linearised Entropy')
        plt.ylabel('Field Linearised Entropy')

        # plt.subplot(6, 1, 2)
        # plt.plot(np.arange(i_trials + 1), entropy_monte_carlo_array[:(i_trials + 1)])
        # plt.title('Field True Entropy through Monte Carlo')
        # plt.ylabel('Field True Entropy')

        plt.subplot(5, 1, 2)
        plt.plot(np.arange(i_trials + 1), entropy_linearised_mean_array[:(i_trials + 1)])
        plt.title('Average Linearised Entropy')
        plt.ylabel('Average Linearised Entropy')

        plt.subplot(5, 1, 3)
        plt.plot(np.arange(i_trials + 1), entropy_true_mean_array[:(i_trials + 1)])
        plt.title('Average True Entropy')
        plt.ylabel('Average True Entropy')

        plt.subplot(5, 1, 4)
        plt.plot(np.arange(i_trials + 1), entropy_opt_array[:(i_trials + 1)])
        plt.title('Joint entropy of path chosen each iteration')
        plt.xlabel('Steps')
        plt.ylabel('Joint Entropy')

        plt.subplot(5, 1, 5)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.plot(np.arange(i_trials + 1), 100 * mistake_ratio_array[:(i_trials + 1)])
        plt.title('Prediction Miss Ratio')
        plt.xlabel('Steps')
        plt.ylabel('Prediction Miss Ratio (%)')
        
        # Save the plot
        plt.savefig('%sentropy_history%d.png' 
            % (full_directory, i_trials + 1))
        logging.info('Plotted and Saved Iteration')

        # Move on to the next step
        i_trials += 1

        # Save the learned classifier
        if i_trials % 50 == 0:
            np.savez('%slearned_classifier_trial%d.npz'
                % (full_directory, i_trials), 
                learned_classifier = learned_classifier)


    # When finished, save the learned classifier
    np.savez('%slearned_classifier_final.npz' % full_directory, 
        learned_classifier = learned_classifier)

    # Show everything!
    plt.show()

def keep_going_until_surprise(xq_abs_opt, learned_classifier, decision_boundary):

    if isinstance(learned_classifier, list):
        y_unique = learned_classifier[0].cache.get('y_unique')
    else:
        y_unique = learned_classifier.cache.get('y_unique')

    yq_pred = gp.classifier.classify(gp.classifier.predict(xq_abs_opt, 
            learned_classifier), y_unique)

    yq_true = gp.classifier.utils.make_decision(xq_abs_opt, decision_boundary)

    assert yq_pred.dtype == int
    assert yq_true.dtype == int
    k_step = int(np.arange(yq_pred.shape[0])[yq_pred != yq_true].min()) + 1
    if not isinstance(k_step, int):
        logging.debug('"k_step" is not an integer but instead is {0}'.format(k_step))
        k_step = 1
    if not k_step > 0:
        logging.debug('"k_step" is not positive but instead is {0}'.format(k_step))
        k_step = 1

    if k_step > yq_pred.shape[0]/2:
        k_step = int(yq_pred.shape[0]/2)
    return k_step

def initiate_with_continuity(theta_add_opt, k_step = 1):

    theta_add_next = np.zeros(theta_add_opt.shape)

    theta_add_next[0] = theta_add_opt[:(k_step + 1)].sum() % (2 * np.pi)
    theta_add_next[1:-k_step] = theta_add_opt[(k_step + 1):]

    return theta_add_next

def boundary_map(Xq): 

    test_range_min = -2.0
    test_range_max = +2.0
    return True if  np.any(Xq[:, 0] < test_range_min) | \
                    np.any(Xq[:, 0] > test_range_max) | \
                    np.any(Xq[:, 1] < test_range_min) | \
                    np.any(Xq[:, 1] > test_range_max)   \
                else False


def forward_path_model(theta_add, r, x):

    theta = np.cumsum(theta_add)

    x1_add = r * np.cos(theta)
    x2_add = r * np.sin(theta)

    x1_rel = np.cumsum(x1_add)
    x2_rel = np.cumsum(x2_add)

    x_rel = np.array([x1_rel, x2_rel]).T

    Xq = x + x_rel
    return Xq

def path_linearised_entropy_model(theta_add, r, x, memory):

    Xq = forward_path_model(theta_add, r, x)
    
    logging.info('Computing linearised entropy...')
    start_time = time.clock()
    predictors = gp.classifier.query(memory, Xq)
    yq_exp = gp.classifier.expectance(memory, predictors)
    yq_cov = gp.classifier.covariance(memory, predictors)
    entropy = gp.classifier.linearised_entropy(yq_exp, yq_cov, memory)
    logging.debug('Linearised entropy computational time : %.8f' % 
        (time.clock() - start_time))

    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_add), entropy))
    return entropy

def path_monte_carlo_entropy_model(theta_add, r, x, memory, 
    n_draws = 1000, S = None):

    Xq = forward_path_model(theta_add, r, x)

    logging.info('Computing monte carlo joint entropy...')
    start_time = time.clock()
    predictors = gp.classifier.query(memory, Xq)
    yq_exp = gp.classifier.expectance(memory, predictors)
    yq_cov = gp.classifier.covariance(memory, predictors)
    entropy = gp.classifier.monte_carlo_joint_entropy(yq_exp, yq_cov, memory, 
        n_draws = n_draws, S = S, processes = 1)
    logging.debug('Monte carlo joint entropy computational time : %.8f' % 
        (time.clock() - start_time))

    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_add), entropy))
    return entropy

def path_bounds_model(theta_add, r, x, ranges):

    Xq = forward_path_model(theta_add, r, x)

    # Assume ranges is symmetric (a square)
    c = np.max(np.abs(Xq)) - ranges[1]
    logging.debug('Contraint Violation: %.5f' % c)
    return c

def go_optimised_path(theta_add_init, x, r, memory, ranges,
    theta_add_low = None, theta_add_high = None, walltime = None, 
    xtol_rel = 1e-2, ftol_rel = 1e-2, globalopt = False, objective = 'LE',
    n_draws = 5000):

    ##### OPTIMISATION #####
    try:

        if objective == 'LE':

            def objective(theta_add, grad):
                return path_linearised_entropy_model(theta_add, r, x, memory)

        elif objective == 'MCJE':

            S = np.random.normal(loc = 0., scale = 1., 
                size = (theta_add_init.shape[0], n_draws))

            def objective(theta_add, grad):
                return path_monte_carlo_entropy_model(theta_add, r, x, memory, 
                    n_draws = n_draws, S = S)

        def constraint(theta_add, grad):
            return path_bounds_model(theta_add, r, x, ranges)

        n_params = theta_add_init.shape[0]

        if globalopt:

            opt = nlopt.opt(nlopt.G_MLSL_LDS, n_params)
            local_opt = nlopt.opt(nlopt.LN_COBYLA , n_params)
            opt.set_local_optimizer(local_opt)

        else:

            opt = nlopt.opt(nlopt.LN_COBYLA , n_params)


        opt.set_lower_bounds(theta_add_low)
        opt.set_upper_bounds(theta_add_high)
        opt.set_maxtime(walltime)

        if xtol_rel:
            opt.set_xtol_rel(xtol_rel)

        if ftol_rel:
            opt.set_ftol_rel(ftol_rel)
        

        opt.set_max_objective(objective)
        opt.add_inequality_constraint(constraint, 1e-2)

        theta_add_opt = opt.optimize(theta_add_init)

        entropy_opt = opt.last_optimum_value()

    except Exception as e:

        theta_add_opt = initiate_with_continuity(theta_add_init)
        entropy_opt = np.nan
        logging.warning('Problem with optimisation. Continuing planned route.')
        logging.warning(type(e))
        logging.warning(e)
        logging.debug('Initial parameters: {0}'.format(theta_add_init))

    ##### PATH COMPUTATION #####
    x_abs_opt = forward_path_model(theta_add_opt, r, x)

    return x_abs_opt, theta_add_opt, entropy_opt

def generate_line_path(x_s, x_f, n_points = 10):
    p = x_f - x_s

    r = np.linspace(0, 1, num = n_points)
    return np.outer(r, p) + x_s

def generate_line_paths(X_s, X_f, n_points = 10):

    assert X_s.shape == X_f.shape

    if hasattr(n_points, '__iter__'):

        assert n_points.shape[0] == X_s.shape[0]
        X = np.array([generate_line_path(X_s[i], X_f[i], n_points[i]) for i in range(X_s.shape[0])])
        return X.reshape(X.shape[0] * X.shape[1], X.shape[2])

    else:

        X = np.array([generate_line_path(X_s[i], X_f[i], n_points) for i in range(X_s.shape[0])])
        return X.reshape(X.shape[0] * X.shape[1], X.shape[2])

if __name__ == "__main__":
    main()

# DO TO: Put learned hyperparam in the title
# DO TO: Find joint entropy of the whole region and put it in the title
# TO DO: Find other measures of improvement (sum of entropy (linearised and true)) (sum of variances and standard deviation)