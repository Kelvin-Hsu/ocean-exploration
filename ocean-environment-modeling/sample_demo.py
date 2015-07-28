"""
Demonstration of simple exploration algorithms using generated data

Author: Kelvin
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import computers.gp as gp
import computers.unsupervised.whitening as pre
import sys
import logging
import shutil
import nlopt

# Talk about why linearising keeps the squashed probability 
# distributed as a gaussian (yes, the probability itself is a random variable)

plt.ion()

# Define the kernel used for classification
def kerneldef(h, k):
    return h(1e-3, 1e5, 10) * k('gaussian', 
                                [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])

def main():

    """
    Demostration Options
    """
    logging.basicConfig(level = logging.DEBUG)

    # If using parallel functionality, you must call this to set the appropriate
    # logging level
    gp.classifier.set_multiclass_logging_level(logging.DEBUG)

    # np.random.seed(220)
    # Feature Generation Parameters and Demonstration Options
    SAVE_OUTPUTS = True # We don't want to make files everywhere for a demo.
    SHOW_RAW_BINARY = True
    test_range_min = -2.5
    test_range_max = +2.5
    n_train = 200
    n_query = 250
    n_dims  = 2   # <- Must be 2 for vis
    n_cores = None # number of cores for multi-class (None -> default: c-1)
    walltime = 300.0
    approxmethod = 'laplace' # 'laplace' or 'pls'
    multimethod = 'OVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
    responsename = 'probit' # 'probit' or 'logistic'
    batch_start = False
    entropy_threshold = None
    mycmap = cm.jet

    n_draws = 6
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
    db8 = lambda x1, x2: (np.cos(x1**2 + x2) + np.sin(x1 + x2**2) > 0) | (((x1 - 1)**2 + x2**2/4) * 
            (0.9*(x1 + 1)**2 + x2**2/2) < 1.4) & \
            ((x1 + x2) < 1.5) | (x1 < -1.9) | (x1 > +1.9) | (x2 < -1.9) | (x2 > +1.9) | ((x1 + 0.75)**2 + (x2 - 1.5)**2 < 0.3**2)
    # db9 = lambda x1, x2: ((x1)**2 + (x2)**2 < 0.3**2) | ((x1)**2 + (x2)**2 > 0.5**2) |
    decision_boundary  = db4a # [db5a, db5b, db1a, db5, db4a]

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
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    # Query Points
    Xq = np.random.uniform(test_range_min, test_range_max, 
        size = (n_query, n_dims))
    xq1 = Xq[:, 0]
    xq2 = Xq[:, 1]

    # Training Labels
    y = gp.classifier.utils.make_decision(X, decision_boundary)
    y_unique = np.unique(y)

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
        train = True, ftol = 1e-10, processes = n_cores)

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

    # Obtain the prediction function
    prediction_function = lambda Xq: gp.classifier.predict(Xq, 
                            learned_classifier,
                            fusemethod = fusemethod, processes = n_cores)
    # Prediction
    yq_prob = prediction_function(Xq)
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

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        THE GAP BETWEEN ANALYSIS AND PLOTS
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""







    """
    Classifier Prediction Results (Plots)
    """

    print('Plotting... please wait')

    """
    Plot: Training Set
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
    print('Plotted Training Set')

    """
    Plot: Prediction Probabilities
    """

    # Visualise the prediction probabilities
    (yq_prob_plt, Xq_plt, _, _, _) = \
        gp.classifier.utils.visualise_prediction_probabilities_multiclass(
        test_range_min, test_range_max, prediction_function, y)

    # To avoid re-computing most of the results for plotting, 
    # find the entropy and predictions here
    yq_entropy_plt = gp.classifier.entropy(yq_prob_plt)
    yq_pred_plt = gp.classifier.classify(yq_prob_plt, y)
    print('Plotted Prediction Probabilities')

    """
    Plot: Prediction Labels
    """

    # Query (Prediction Map)
    fig = plt.figure()
    gp.classifier.utils.visualise_prediction(test_range_min, test_range_max, 
        lambda Xq: gp.classifier.classify(prediction_function(Xq), y), 
        cmap = mycmap, yq_pred = yq_pred_plt)

    plt.title('Prediction')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    print('Plotted Prediction Labels')

    """
    Plot: Prediction Entropy onto Training Set
    """

    # Query (Entropy) and Training Set
    fig = plt.figure()
    gp.classifier.utils.visualise_entropy(test_range_min, test_range_max, 
        lambda Xq: gp.classifier.entropy(prediction_function(Xq)), 
        entropy_threshold = entropy_threshold, yq_entropy = yq_entropy_plt)
    plt.title('Prediction Entropy and Training Set')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    print('Plotted Prediction Entropy on Training Set')

    """
    Plot: Linearised Prediction Entropy onto Training Set
    """

    logging.info('Caching Predictor...')
    predictor_plt = gp.classifier.query(learned_classifier, Xq_plt)
    logging.info('Computing Expectance...')
    expectance_latent_plt = gp.classifier.expectance(learned_classifier, 
        predictor_plt)
    logging.info('Computing Variance...')
    variance_latent_plt = gp.classifier.variance(learned_classifier, 
        predictor_plt)
    logging.info('Computing Linearised Entropy...')
    entropy_latent_plt = gp.classifier.linearised_entropy(expectance_latent_plt,
        variance_latent_plt, learned_classifier)

    # Query (Entropy) and Training Set
    fig = plt.figure()
    gp.classifier.utils.visualise_entropy(test_range_min, test_range_max, 
        lambda Xq: gp.classifier.entropy(prediction_function(Xq)), 
        entropy_threshold = entropy_threshold, yq_entropy = entropy_latent_plt)
    plt.title('Linearised Prediction Entropy and Training Set')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    print('Plotted Prediction Entropy on Training Set')

    """
    Plot: Prediction Probabilities from Binary Classifiers
    """    

    # Query (Individual Binary CLassifier Maps)
    if SHOW_RAW_BINARY:
        gp.classifier.utils.reveal_binary(test_range_min, test_range_max, 
            gp.classifier.predict, learned_classifier, y_unique)

    """
    Plot: Sample Query Predictions
    """  

    # Visualise Predictions
    fig = plt.figure()
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
    print('Plotted Sample Query Labels')

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
        print('Plotted Sample Query Draws')

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
        shutil.copy2('./sampledemo.py', full_directory)


    print('Modeling Done')

    """
    Path Planning
    """

    # Pick a starting point
    xq_start = np.array([0.0, 0.0])

    # The directions the vehicle can travel
    n_paths = 15
    thetas = np.linspace(0, 2*np.pi, 
        num = n_paths + 1)[:-1][:, np.newaxis][:, np.newaxis]

    # The steps the vehicle can travel
    horizon = 0.6
    n_steps = 6
    steps = np.linspace(horizon/n_steps, horizon, num = n_steps)[:, np.newaxis]

    # Where the vehicle is right now
    xq_now = xq_start.copy()

    # Compose the road that extends radially from the center
    Xq_roads = compose_roads(thetas, steps)

    # The observed data till now
    X_now = X.copy()
    y_now = y.copy()

    # Observe the current location
    Xq_now = np.array([xq_now])
    yq_now = gp.classifier.utils.make_decision(Xq_now, decision_boundary)

    # Add the observed data to the training set
    X_now = np.concatenate((X_now, Xq_now), axis = 0)
    y_now = np.append(y_now, yq_now)

    # Add the new location to the array of travelled coordinates
    xq1_nows = np.array([xq_now[0]])
    xq2_nows = np.array([xq_now[1]])
    yq_nows = np.array([yq_now])

    # Plot the current situation
    fig1 = plt.figure(figsize = (15, 15))
    # plt.scatter(xq_now[0], xq_now[1], c = yq_now, s = 60, 
    #     vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)


    fig2 = plt.figure(figsize = (15, 15))
    # plt.scatter(xq_now[0], xq_now[1], c = yq_now, s = 60, 
    #     vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)

    # Start exploring
    i_trials = 0
    i_path_chosen = None
    while i_trials < 1001:

        # Train the classifier!
        logging.info('Learning Classifier...')
        batch_config = gp.classifier.batch_start(optimiser_config, 
            learned_classifier)
        try:
            learned_classifier = gp.classifier.learn(X_now, y_now, kerneldef,
                responsefunction, batch_config, 
                multimethod = multimethod, approxmethod = approxmethod,
                train = True, ftol = 1e-4, processes = n_cores)
        except:
            learned_classifier = gp.classifier.learn(X_now, y_now, kerneldef,
                responsefunction, batch_config, 
                multimethod = multimethod, approxmethod = approxmethod,
                train = False, ftol = 1e-4, processes = n_cores)            

        logging.info('Caching Predictor...')
        predictor_plt = gp.classifier.query(learned_classifier, Xq_plt)
        logging.info('Computing Expectance...')
        expectance_latent_plt = gp.classifier.expectance(learned_classifier, 
            predictor_plt)
        logging.info('Computing Variance...')
        variance_latent_plt = gp.classifier.variance(learned_classifier, 
            predictor_plt)
        logging.info('Computing Linearised Entropy...')
        entropy_latent_plt = gp.classifier.linearised_entropy(expectance_latent_plt,
            variance_latent_plt, learned_classifier)

        if i_trials >= 0:
            vmin = entropy_latent_plt.min()
            vmax = entropy_latent_plt.max()
            # vmax = np.array([entropy_latent_plt.max(), 0]).max()

        plt.figure(fig1.number)
        plt.clf()
        plt.title('Exploration track and linearised entropy [horizon = %.2f]' % horizon)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))

        # Query (Entropy) and Training Set
        gp.classifier.utils.visualise_entropy(test_range_min, test_range_max, 
                None, entropy_threshold = entropy_threshold, 
                yq_entropy = entropy_latent_plt, vmin = vmin, vmax = vmax)
        plt.colorbar()
        plt.scatter(x1, x2, c = y, s = 40, marker = 'x', cmap = mycmap)

        # Obtain where to query
        Xq_roads_abs = xq_now + Xq_roads
        Xq_now_paths = Xq_roads_abs.reshape(thetas.shape[0], steps.shape[0], 
            n_dims)

        # Pick the point towards the highest entropy path
        (xq_now, i_path_chosen) = \
            go_highest_entropy_path(Xq_now_paths, learned_classifier, i_path_last = i_path_chosen)

        # r = 0.1
        # choice_walltime = 60.0
        # ftol_rel = 1e-2
        # go_optimised_path(theta_add_init, learned_classifier, xq_now, r, theta_add_low = theta_add_low, theta_add_high = theta_add_high, walltime = choice_walltime, ftol_rel = ftol_rel)

        # Observe the current location
        Xq_now = np.array([xq_now])
        yq_now = gp.classifier.utils.make_decision(Xq_now, decision_boundary)

        # Add the observed data to the training set
        X_now = np.concatenate((X_now, Xq_now), axis = 0)
        y_now = np.append(y_now, yq_now)

        # Add the new location to the array of travelled coordinates
        xq1_nows = np.append(xq1_nows, xq_now[0])
        xq2_nows = np.append(xq2_nows, xq_now[1])
        yq_nows = np.append(yq_nows, yq_now)

        # Plot the current situation
        plt.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none',
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.scatter(xq_now[0], xq_now[1], c = yq_now, s = 60, 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        gp.classifier.utils.plot_circle(xq_now, horizon, c = 'k', marker = '.')

        # Save the current situation
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sstep%d.png' % (full_directory, i_trials + 1))

        plt.figure(fig2.number)
        plt.clf()

        plt.title('Exploration track and true entropy [horizon = %.2f]' % horizon)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim((test_range_min, test_range_max))
        plt.ylim((test_range_min, test_range_max))

        # Also plot actual entropy
        entropy_plt = gp.classifier.entropy(gp.classifier.predict_from_latent(expectance_latent_plt, 
            variance_latent_plt, learned_classifier, fusemethod = fusemethod))

        if i_trials >= 0:
            vmin2 = entropy_plt.min()
            vmax2 = entropy_plt.max()
            # vmax2 = np.array([entropy_plt.max(), 1]).max()

        # Query (Entropy) and Training Set
        gp.classifier.utils.visualise_entropy(test_range_min, test_range_max, 
                None, entropy_threshold = entropy_threshold, 
                yq_entropy = entropy_plt, vmin = vmin2, vmax = vmax2)
        plt.colorbar()
        plt.scatter(x1, x2, c = y, s = 40, marker = 'x', cmap = mycmap)

        # Plot the current situation
        plt.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none',
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        plt.scatter(xq_now[0], xq_now[1], c = yq_now, s = 60, 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        gp.classifier.utils.plot_circle(xq_now, horizon, c = 'k', marker = '.')

        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sstep_true_entropy%d.png' % (full_directory, i_trials + 1))


        i_trials += 1

        if i_trials % 50 == 0:
            np.savez('%slearned_classifier_trial%d.npz' % (full_directory, i_trials), 
                learned_classifier = learned_classifier)

    # Show everything!
    plt.show()

def forward_path_model(theta_add, r, x, memory):

    theta = np.cumsum(theta_add)

    x1_add = r * np.cos(theta)
    x2_add = r * np.sin(theta)

    x1_rel = np.cumsum(x1_add)
    x2_rel = np.cumsum(x2_add)

    # x_rel = np.concatenate((x1_rel[:, np.newaxis], x2_rel[:, np.newaxis]), axis = 1)
    x_rel = np.array([x1_rel, x2_rel]).T

    Xq = x + x_rel

    try:
        logging.info('Caching Predictor...')
        predictors = gp.classifier.query(memory, Xq)
        logging.info('Computing Expectance...')
        yq_exp = gp.classifier.expectance(memory, predictors)
        logging.info('Computing Covariance...')
        yq_cov = gp.classifier.covariance(memory, predictors)
        logging.info('Computing Entropy...')
        entropy = gp.classifier.linearised_entropy(yq_exp, yq_cov, memory)
    except:
        logging.warning('Failed to compute linearised entropy')
        entropy = -np.inf

    return entropy

def go_optimised_path(theta_add_init, memory, x, r, theta_add_low = None, theta_add_high = None, walltime = None, ftol_rel = 1e-2):

    ##### OPTIMISATION #####
    objective = lambda theta_add: forward_path_model(theta_add, r, x, memory)

    n_params = theta_add_init.shape[0]

    opt = nlopt.opt(nlopt.LN_BOBYQA, n_params)

    if theta_add_low:
        opt.set_lower_bounds(theta_add_low)

    if theta_add_high:
        opt.set_upper_bounds(theta_add_high)
    
    if walltimeL:
        opt.set_maxtime(walltime)

    if ftol_rel:
        opt.set_ftol_rel(ftol_rel)

    opt.set_max_objective(objective)

    theta_add_opt = opt.optimize(theta_add_init)

    entropy_opt = opt.last_optimum_value()

    theta_opt = np.cumsum(theta_add_opt)

    x1_add_opt = r * np.cos(theta_opt)
    x2_add_opt = r * np.sin(theta_opt)

    x1_rel_opt = np.cumsum(x1_add_opt)
    x2_rel_opt = np.cumsum(x2_add_opt)

    x_rel_opt = np.array([x1_rel_opt, x2_rel_opt]).T

    x_abs_opt = x + x_rel_opt

    return x_abs_opt, theta_add_opt, entropy_opt

def restrain_paths(i_path_last, n_paths, n_restrain):

    if i_path_last is not None:
        assert n_restrain % 2 == 1
        n_side = int((n_restrain - 1)/2)
        i_paths_restrained = \
            (i_path_last + np.arange(-n_side, n_side + 1)) % n_paths
    else:
        i_paths_restrained = np.arange(n_paths)

    return i_paths_restrained

# def go_regional_entropy_reduction(x, Xq_region, Xq_directions, n_steps, d_step, learned_classifier, i_path_last = None):

#     (n_paths, n_dims) = Xq_directions.shape

#     Xq_consider = x + Xq_directions

    









#     # The array store the regional entropy for each path
#     yq_region_entropy = -np.inf * np.ones(n_paths)

#     # Restrain the AUV to turn smoothly
#     n_restrain = 5
#     i_paths_restrained = restrain_paths(i_path_last, n_paths, n_restrain)

#     # For each region, compute the joint entropy for that region
#     for i_paths in i_paths_restrained:

#         try:
#             logging.info('Caching Predictor...')
#             predictors = gp.classifier.query(learned_classifier, 
#                 Xq_paths[i_paths])
#             logging.info('Computing Expectance...')
#             yq_exp = gp.classifier.expectance(learned_classifier, predictors)
#             logging.info('Computing Covariance...')
#             yq_cov = gp.classifier.covariance(learned_classifier, predictors)
#             logging.info('Computing Entropy...')
#             yq_region_entropy[i_paths] = \
#                 gp.classifier.linearised_entropy(yq_exp, yq_cov, learned_classifier)
#         except:
#             yq_region_entropy[i_paths] = -np.inf
#         logging.info(i_paths)

#     # Find the path of maximum joint entropy
#     i_path_max_entropy = yq_region_entropy.argmax()

#     # Move towards that path
#     xq_next = Xq_paths[i_path_max_entropy, 0, :]
#     return xq_next, i_path_max_entropy

def go_highest_entropy_path(Xq_now_paths, learned_classifier, i_path_last = None):

    # Xq_now_paths.shape = (n_paths, n_steps, n_dims)
    n_paths = Xq_now_paths.shape[0]

    # The array store the entropy for each path
    yq_path_entropy = -np.inf * np.ones(n_paths)

    if i_path_last is not None:
        n_path_to_check = 7 # must be odd 
        assert n_path_to_check % 2 == 1
        side = int((n_path_to_check - 1)/2)
        i_path_to_check = (i_path_last + np.arange(-side, side + 1)) % n_paths
    else:
        i_path_to_check = np.arange(n_paths)

    # For each region, compute the joint entropy for that region
    for i_paths in i_path_to_check:

        try:
            logging.info('Caching Predictor...')
            predictors = gp.classifier.query(learned_classifier, 
                Xq_now_paths[i_paths])
            logging.info('Computing Expectance...')
            yq_exp = gp.classifier.expectance(learned_classifier, predictors)
            logging.info('Computing Covariance...')
            yq_cov = gp.classifier.covariance(learned_classifier, predictors)
            logging.info('Computing Entropy...')
            yq_path_entropy[i_paths] = \
                gp.classifier.linearised_entropy(yq_exp, yq_cov, learned_classifier)
        except:
            yq_region_entropy[i_paths] = -np.inf
        logging.info(i_paths)

    # Find the path of maximum joint entropy
    i_path_max_entropy = yq_path_entropy.argmax()

    # Move towards that path
    xq_next = Xq_now_paths[i_path_max_entropy, 0, :]
    return xq_next, i_path_max_entropy

def go_highest_entropy_region(Xq_now_paths, learned_classifier, i_path_last = None):

    # Xq_now_paths.shape = (n_paths, n_steps, n_dims)
    (n_paths, n_steps, n_dims) = Xq_now_paths.shape

    # The array store the entropy for each region
    yq_region_entropy = -np.inf * np.ones(n_paths)

    if i_path_last is not None:
        n_path_to_check = 9 # must be odd 
        assert n_path_to_check % 2 == 1
        side = int((n_path_to_check - 1)/2)
        i_path_to_check = (i_path_last + np.arange(-side, side + 1)) % n_paths
    else:
        i_path_to_check = np.arange(n_paths)

    # For each region, compute the joint entropy for that region
    for i_paths in i_path_to_check:

        try:
            Xq_now_region = Xq_now_paths[(i_paths + np.arange(-1, 2)) % 3].reshape(3 * n_steps, 2)

            logging.info('Caching Predictor...')
            predictors = gp.classifier.query(learned_classifier, Xq_now_region)
            logging.info('Computing Expectance...')
            yq_exp = gp.classifier.expectance(learned_classifier, predictors)
            logging.info('Computing Covariance...')
            yq_cov = gp.classifier.covariance(learned_classifier, predictors)
            logging.info('Computing Entropy...')
            yq_region_entropy[i_paths] = \
                gp.classifier.linearised_entropy(yq_exp, yq_cov, learned_classifier)
        except:
            yq_region_entropy[i_paths] = -np.inf
        logging.info(i_paths)

    # Find the region of maximum joint entropy
    i_path_max_entropy = yq_region_entropy.argmax()

    # Move towards that region
    xq_next = Xq_now_paths[i_path_max_entropy, 0, :]
    return xq_next, i_path_max_entropy

def compose_roads(thetas, steps):

    # Compose a road structure spanning radially outwards
    n_dims = 2
    Xq_hyper = np.zeros((thetas.shape[0], steps.shape[0], n_dims))
    Xq_hyper[:, :, [0]] = steps * np.cos(thetas)
    Xq_hyper[:, :, [1]] = steps * np.sin(thetas)
    Xq_roads = Xq_hyper.reshape(thetas.shape[0] * steps.shape[0], n_dims)
    return Xq_roads

if __name__ == "__main__":
    main()