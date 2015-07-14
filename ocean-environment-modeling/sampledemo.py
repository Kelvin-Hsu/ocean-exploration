"""
Test 'generalClassifier'
Demonstration of general gp classifiers
Depending on the number of unique labels, this performs either binary 
classification or multiclass classification using binary classifiers 
(AVA or OVA)

Author: Kelvin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import computers.gp as gp
import computers.unsupervised.whitening as pre
import sys
import logging


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

    # np.random.seed(200)
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
            initial_hyperparams = [ [70.8780, 0.754, 0.462],  \
                                    [152.485, 0.737, 0.623], \
                                    [781.625, 1.395, 1.678], \
                                    [455.537, 1.477, 0.767]]
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
    # logging.info('Computing Group Entropy from GP...')
    # group_entropy_latent = gp.classifier.entropy_latent(yq_cov_list[0])
    # print(group_entropy_latent)
    # return

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
    logging.info('Compuating Variance...')
    variance_latent_plt = gp.classifier.variance(learned_classifier, predictor_plt)
    logging.info('Computing Latent Entropy...')
    entropy_latent_plt = gp.classifier.entropy_latent(variance_latent_plt)

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
    fig = plt.figure()
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

    xq_start = np.array([-1.5, -1.5])
    thetas = np.linspace(0, 2*np.pi, num = 72 + 1)[:, np.newaxis][:, np.newaxis]
    horizon = 0.5
    steps = np.linspace(0, horizon, num = 10 + 1)[:, np.newaxis]

    xq_now = xq_start.copy()

    Xq_hyper = np.zeros((thetas.shape[0], steps.shape[0], n_dims))
    Xq_hyper[:, :, [0]] = steps * np.cos(thetas)
    Xq_hyper[:, :, [1]] = steps * np.sin(thetas)
    Xq_extend = Xq_hyper.reshape(thetas.shape[0] * steps.shape[0], n_dims)

    i = 0
    fig = plt.figure()
    while True:

        Xq_now = xq_now + Xq_extend

        yq_prob = prediction_function(Xq_now)
        yq_pred = gp.classifier.classify(yq_prob, y)
        yq_entropy = gp.classifier.entropy(yq_prob)
            
        # xq1_now = Xq_now[:, 0]
        # xq2_now = Xq_now[:, 1]
        # plt.scatter(xq1_now, xq2_now, marker = 'x', cmap = mycmap)
        # plt.draw()
        wait = input('PRESS ENTER TO CONTINUE.')

    # Show everything!
    plt.show()





if __name__ == "__main__":
    main()