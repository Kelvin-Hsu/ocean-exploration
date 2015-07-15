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
import computers.gp as gp
import computers.unsupervised.whitening as pre
import sys
import logging

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

    np.random.seed(200)
    # Feature Generation Parameters and Demonstration Options
    SAVE_OUTPUTS = False # We don't want to make files everywhere for a demo.
    SHOW_RAW_BINARY = False
    test_range_min = -1.75
    test_range_max = +1.75
    n_train = 200
    n_query = 250
    n_dims  = 2   # <- Must be 2 for vis
    n_cores = None # number of cores for multi-class (None -> default: c-1)
    walltime = 300.0
    approxmethod = 'laplace' # 'laplace' or 'pls'
    multimethod = 'OVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
    responsename = 'probit' # 'probit' or 'logistic'
    batch_start = True
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
            (0.9*(x1 + 1)**2 + x2**2/2) > 0.2)
    db3 = lambda x1, x2: ((x1 + x2) < 2) & ((x1 + x2) > -2.2)
    db4 = lambda x1, x2: ((x1 - 0.75)**2 + (x2 + 0.8)**2 > 0.3**2)
    db5 = lambda x1, x2: ((x1/2)**2 + x2**2 > 0.3)
    db6 = lambda x1, x2: (((x1 - 0.5)/8)**2 + (x2 + 1.5)**2 > 0.2**2)
    decision_boundary  = [db1, db3, db5]

    """
    Data Generation
    """

    # Training Points
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
    print('===Begin Classifier Training===')
    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # User can choose to batch start each binary classifier with different
    # initial hyperparameters for faster training
    if batch_start:
        if y_unique.shape[0] == 2:
            initial_hyperparams = [10, 0.1, 0.1]
        elif multimethod == 'OVA':
            initial_hyperparams = [  [356.46828146743388, 0.7628014361167047, 0.53093943834222967], \
                                     [356.55639115411111, 0.8360802461779917, 0.76369284639761281], \
                                     [472.00684107153467, 1.6484025983516082, 1.5502023765326216], \
                                     [239.72087459618402, 1.3077546793647976, 0.72127460091935158] ]
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

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        THE GAP BETWEEN ANALYSIS AND PLOTS
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""







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
    Plot: Latent Prediction Entropy onto Training Set
    """

    logging.info('Caching Predictor...')
    predictor_plt = gp.classifier.query(learned_classifier, Xq_plt)
    logging.info('Compuating Expectance...')
    expectance_latent_plt = gp.classifier.expectance(learned_classifier, 
        predictor_plt)
    logging.info('Compuating Variance...')
    variance_latent_plt = gp.classifier.variance(learned_classifier, 
        predictor_plt)
    logging.info('Computing Latent Entropy...')
    entropy_latent_plt = gp.classifier.joint_entropy(expectance_latent_plt,
        variance_latent_plt, learned_classifier)

    # Query (Entropy) and Training Set
    fig = plt.figure()
    gp.classifier.utils.visualise_entropy(test_range_min, test_range_max, 
        lambda Xq: gp.classifier.entropy(prediction_function(Xq)), 
        entropy_threshold = entropy_threshold, yq_entropy = entropy_latent_plt)
    plt.title('Latent Prediction Entropy and Training Set')
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
            gp.classifier.predict_binary, learned_classifier, y_unique)

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
                method, fusemethod)
        gp.classifier.utils.save_all_figures(save_directory, append_time = True)


    print('Modeling Done')

    """
    Path Planning
    """

    xq_start = np.array([0.0, 0.0])
    thetas = np.linspace(0, 2*np.pi, num = 10 + 1)[:-1][:, np.newaxis][:, np.newaxis]
    horizon = 0.5
    n_steps = 10
    steps = np.linspace(horizon/n_steps, horizon, num = n_steps)[:, np.newaxis]

    xq_now = xq_start.copy()

    Xq_hyper = np.zeros((thetas.shape[0], steps.shape[0], n_dims))
    Xq_hyper[:, :, [0]] = steps * np.cos(thetas)
    Xq_hyper[:, :, [1]] = steps * np.sin(thetas)
    Xq_extend = Xq_hyper.reshape(thetas.shape[0] * steps.shape[0], n_dims)

    X_now = X.copy()
    y_now = y.copy()

    xq1_nows = np.array([])
    xq2_nows = np.array([])

    i = 0
    fig = plt.figure(figsize = (15, 15))
    plt.scatter(xq_now[0], xq_now[1], marker = '+', s = 80, 
        cmap = mycmap)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((test_range_min, test_range_max))
    plt.ylim((test_range_min, test_range_max))
    i_trials = 0
    while i_trials < 100:

        # Train the classifier!
        logging.info('Learning Classifier...')
        batch_config = gp.classifier.batch_start(optimiser_config, 
            learned_classifier)
        learned_classifier = gp.classifier.learn(X_now, y_now, kerneldef,
            responsefunction, batch_config, 
            multimethod = multimethod, approxmethod = approxmethod,
            train = True, ftol = 1e-10, processes = n_cores)

        logging.info('Caching Predictor...')
        predictor_plt = gp.classifier.query(learned_classifier, Xq_plt)
        logging.info('Compuating Expectance...')
        expectance_latent_plt = gp.classifier.expectance(learned_classifier, 
            predictor_plt)
        logging.info('Compuating Variance...')
        variance_latent_plt = gp.classifier.variance(learned_classifier, 
            predictor_plt)
        logging.info('Computing Latent Entropy...')
        entropy_latent_plt = gp.classifier.joint_entropy(expectance_latent_plt,
            variance_latent_plt, learned_classifier)

        # Query (Entropy) and Training Set
        gp.classifier.utils.visualise_entropy(test_range_min, test_range_max, 
                None, entropy_threshold = entropy_threshold, 
                yq_entropy = entropy_latent_plt)
        if i_trials == 0:
            plt.colorbar()
        plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)

        # Obtain where to query
        Xq_now = xq_now + Xq_extend
        Xq_now_paths = Xq_now.reshape(thetas.shape[0], steps.shape[0], n_dims)

        xq_now = go_highest_entropy_path(Xq_now_paths, learned_classifier)

        Xq_now = np.array([xq_now])
        yq_now = gp.classifier.utils.make_decision(Xq_now, decision_boundary)

        X_now = np.concatenate((X_now, Xq_now), axis = 0)
        y_now = np.append(y_now, yq_now)

        xq1_nows = np.append(xq1_nows, xq_now[0])
        xq2_nows = np.append(xq2_nows, xq_now[1])
        plt.scatter(xq1_nows, xq2_nows, c = 'k', marker = '+', s = 80)

        i_trials += 1
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('figure%d.png' % i_trials)

    # Show everything!
    plt.show()


def go_highest_entropy_path(Xq_now_paths, learned_classifier):

    n_paths = Xq_now_paths.shape[0]
    yq_path_entropy = np.zeros(n_paths)
    for i_paths in range(n_paths):

        logging.info('Caching Predictor...')
        predictors = gp.classifier.query(learned_classifier, 
            Xq_now_paths[i_paths])
        logging.info('Computing Expectance...')
        yq_exp = gp.classifier.expectance(learned_classifier, predictors)
        logging.info('Computing Covariance...')
        yq_cov = gp.classifier.covariance(learned_classifier, predictors)
        logging.info('Computing Entropy...')
        yq_path_entropy[i_paths] = \
            gp.classifier.joint_entropy(yq_exp, yq_cov, learned_classifier)
        logging.info(i_paths)

    i_path_max_entropy = yq_path_entropy.argmax()

    xq_next = Xq_now_paths[i_path_max_entropy, 0, :]
    return xq_next

if __name__ == "__main__":
    main()