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

def parse(key, currentvalue, arg = 1):
    """
    Parses the command line arguments
    1. Obtains the expected data type
    2. Checks if key is present
    3. If it is, check if the expected data type is a boolean
    4. If so, set the flag away from default and return
    5. Otherwise, obtain the 'arg'-th parameter after the key
    6. Cast the resulting string into the correct type and return
    (*) This will return the default value if key is not present
    """
    cast = type(currentvalue)
    if key in sys.argv:
        if cast == bool:
            return not currentvalue
        currentvalue = sys.argv[sys.argv.index(key) + arg]
    return cast(currentvalue)

def main():

    """
    Command Argument Parser
    """
    FOLDERNAME = ''.join(sys.argv[1:])
    FILENAME = sys.argv[0]

    METHOD = parse('-method', 'LE')
    START_POINT1 = parse('-start', 0.0, arg = 1)
    START_POINT2 = parse('-start', 0.0, arg = 2)
    SEED = parse('-seed', 100)
    N_ELLIPSE = parse('-e', 20)
    RANGE = parse('-range', 2.0)
    N_CLASS = parse('-classes', 4)
    N_STEPS = parse('-steps', 300)
    TRAIN_SET = parse('-train', 'track')
    N_TRAIN = parse('-ntrain', 150)
    WHITEFN = parse('-whiten', 'pca')
    R_START = parse('-rstart', 1.8)
    R_TRACK = parse('-rtrack', 1.0)
    N_TRACK = parse('-ntrack', 10)
    N_TRACK_POINTS = parse('-ntrackpoints', 15)
    TRACK_PERTURB_DEG = parse('-trackperturb', 5.0)
    CHAOS = parse('-chaos', False)
    SHOW_TRAIN = parse('-st', False)
    SAVE_OUTPUTS = True

    """
    Analysis Options
    """
    # Set logging level
    logging.basicConfig(level = logging.DEBUG)
    gp.classifier.set_multiclass_logging_level(logging.DEBUG)

    # Feature Generation Parameters and Demonstration Options
    range_min = -RANGE
    range_max = +RANGE
    ranges = (range_min, range_max)
    n_dims  = 2 # <- Must be 2 for vis
    n_cores = 1 # number of cores for multi-class (None -> default: c-1)
    walltime = 300.0
    approxmethod = 'laplace' # 'laplace' or 'pls'
    multimethod = 'OVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
    responsename = 'probit' # 'probit' or 'logistic'
    n_draws_est = 2500
    meas_points = 10

    np.random.seed(SEED)
    decision_boundary = \
        gp.classifier.utils.generate_elliptical_decision_boundaries(ranges, 
        min_size = 0.1, max_size = 0.5, 
        n_class = N_CLASS, n_ellipse = N_ELLIPSE, n_dims = n_dims)

    if (WHITEFN == 'none') or (WHITEFN == 'NONE'):
        whitenfn = rh.nowhitenfn
    elif (WHITEFN == 'pca') or (WHITEFN == 'PCA'):
        whitenfn = pre.whiten
    elif (WHITEFN == 'standardise') or (WHITEFN == 'STANDARDISE'):
        whitenfn = pre.standardise

    """
    Plot Options
    """
    fontsize = 24
    axis_tick_font_size = 14
    plt_points = 250

    """
    Data Generation
    """
    if TRAIN_SET == 'track':
        X = rh.utils.generate_tracks(R_START, R_TRACK, N_TRACK, N_TRACK_POINTS, 
            perturb_deg_scale = TRACK_PERTURB_DEG)
    elif 'unif'in TRAIN_SET:
        X = np.random.uniform(range_min, range_max, size = (N_TRAIN, n_dims))
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    Xw, whitenparams = whitenfn(X)

    n_train = X.shape[0]
    logging.info('Training Points: %d' % n_train)

    # Training Labels
    y = gp.classifier.utils.make_decision(X, decision_boundary)
    y_unique = np.unique(y)
    assert y_unique.shape[0] == N_CLASS

    if y_unique.shape[0] == 2:
        mycmap = cm.get_cmap(name = 'bone', lut = None)
        mycmap2 = cm.get_cmap(name = 'BrBG', lut = None)
    else:
        mycmap = cm.get_cmap(name = 'gist_rainbow', lut = None)
        mycmap2 = cm.get_cmap(name = 'gist_rainbow', lut = None)
    """
    Classifier Training
    """
    if SHOW_TRAIN:
        fig = plt.figure()
        gp.classifier.utils.visualise_decision_boundary(
            range_min, range_max, decision_boundary)
        plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
        plt.title('Training Labels')
        plt.xlabel('x1')
        plt.ylabel('x2')
        cbar = plt.colorbar()
        cbar.set_ticks(y_unique)
        cbar.set_ticklabels(y_unique)
        plt.xlim((range_min, range_max))
        plt.ylim((range_min, range_max))
        plt.gca().patch.set_facecolor('gray')
        logging.info('Plotted Training Set')
        plt.show()

    # Set optimiser configuration and obtain the response function
    logging.info('===Begin Classifier Training===')
    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime
    responsefunction = gp.classifier.responses.get(responsename)

    # Train the classifier!
    learned_classifier = gp.classifier.learn(Xw, y, kerneldef,
        responsefunction, optimiser_config, 
        multimethod = multimethod, approxmethod = approxmethod,
        train = True, ftol = 1e-6, processes = n_cores)

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
    Xq_plt = gp.classifier.utils.query_map(ranges, n_points = plt_points)
    Xq_meas = gp.classifier.utils.query_map(ranges, n_points = meas_points)
    Xqw_plt = whitenfn(Xq_plt, whitenparams)
    Xqw_meas = whitenfn(Xq_meas, whitenparams)
    yq_truth_plt = gp.classifier.utils.make_decision(Xq_plt, decision_boundary)

    logging.info('Plot: Caching Predictor...')
    predictor_plt = gp.classifier.query(learned_classifier, Xqw_plt)
    logging.info('Plot: Computing Expectance...')
    exp_plt = gp.classifier.expectance(learned_classifier, predictor_plt)
    logging.info('Plot: Computing Variance...')
    var_plt = gp.classifier.variance(learned_classifier, predictor_plt)

    logging.info('Plot: Computing Linearised Entropy...')
    yq_le_plt = gp.classifier.linearised_entropy(exp_plt, var_plt, 
        learned_classifier)
    logging.info('Plot: Computing Equivalent Standard Deviation...')
    eq_sd_plt = gp.classifier.equivalent_standard_deviation(yq_le_plt)
    logging.info('Plot: Computing Prediction Probabilities...')
    yq_prob_plt = gp.classifier.predict_from_latent(exp_plt, var_plt, 
        learned_classifier, fusemethod = fusemethod)
    logging.info('Plot: Computing Information Entropy...')
    yq_entropy_plt = gp.classifier.entropy(yq_prob_plt)
    logging.info('Plot: Computing Class Predicitons...')
    yq_pred_plt = gp.classifier.classify(yq_prob_plt, y_unique)

    logging.info('Measure: Caching Predictor...')
    predictor_meas = gp.classifier.query(learned_classifier, Xqw_meas)
    logging.info('Measure: Computing Expectance...')
    exp_meas = gp.classifier.expectance(learned_classifier, predictor_meas)
    logging.info('Measure: Computing Covariance...')
    cov_meas = gp.classifier.covariance(learned_classifier, predictor_meas)
    logging.info('Measure: Computing Mistake Ratio...')
    mistake_ratio = (yq_truth_plt - yq_pred_plt).nonzero()[0].shape[0] / \
        yq_truth_plt.shape[0]
    logging.info('Measure: Computing Linearised Joint Entropy...')
    start_time = time.clock()
    entropy_linearised_meas = gp.classifier.linearised_entropy(
        exp_meas, cov_meas, learned_classifier)
    logging.info('Computation took %.4f seconds' % (time.clock() - start_time))
    logging.info('Linearised Joint Entropy: %.4f' % entropy_linearised_meas)
    logging.info('Measure: Computing Average Linearised Entropy...')
    entropy_linearised_mean_meas = yq_le_plt.mean()
    logging.info('Measure: Computing Average Information Entropy...')
    entropy_true_mean_meas = yq_entropy_plt.mean()

    """Plot: Ground Truth"""
    fig = plt.figure(figsize = (15 * 1.5, 15))

    plt.subplot(2, 3, 1)
    gp.classifier.utils.visualise_map(yq_truth_plt, ranges, cmap = mycmap)
    plt.title('Ground Truth', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    gp.classifier.utils.visualise_decision_boundary(
        range_min, range_max, decision_boundary)
    logging.info('Plotted Prediction Labels')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    """Plot: Training Set"""
    plt.subplot(2, 3, 2)
    gp.classifier.utils.visualise_decision_boundary(
        range_min, range_max, decision_boundary)
    
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.title('Training Labels', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    cbar = plt.colorbar()
    cbar.set_ticks(y_unique)
    cbar.set_ticklabels(y_unique)
    plt.xlim((range_min, range_max))
    plt.ylim((range_min, range_max))
    plt.gca().patch.set_facecolor('gray')
    logging.info('Plotted Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    """Plot: Prediction Labels"""
    plt.subplot(2, 3, 3)
    gp.classifier.utils.visualise_map(yq_pred_plt, ranges, 
        boundaries = True, cmap = mycmap)
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
    gp.classifier.utils.visualise_map(yq_entropy_plt, ranges, 
        cmap = cm.coolwarm)
    plt.title('Prediction Information Entropy', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((range_min, range_max))
    plt.ylim((range_min, range_max))
    logging.info('Plotted Prediction Information Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """Plot: Linearised Differential Entropy onto Training Set"""
    plt.subplot(2, 3, 5)
    yq_le_plt_min = yq_le_plt.min()
    yq_le_plt_max = yq_le_plt.max()
    gp.classifier.utils.visualise_map(yq_le_plt, ranges, 
        cmap = cm.coolwarm, 
        vmin = -yq_le_plt_max, vmax = yq_le_plt_max)
    plt.title('Linearised Differential Entropy', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((range_min, range_max))
    plt.ylim((range_min, range_max))
    logging.info('Plotted Linearised Differential Entropy on Training Set')
    plt.gca().set_aspect('equal', adjustable = 'box')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
        
    """Plot: Equivalent Standard Deviation onto Training Set"""
    plt.subplot(2, 3, 6)
    gp.classifier.utils.visualise_map(eq_sd_plt, ranges, cmap = cm.coolwarm)
    plt.title('Equivalent Standard Deviation', fontsize = fontsize)
    plt.xlabel('x1', fontsize = fontsize)
    plt.ylabel('x2', fontsize = fontsize)
    plt.colorbar()
    plt.scatter(x1, x2, c = y, marker = 'x', cmap = mycmap)
    plt.xlim((range_min, range_max))
    plt.ylim((range_min, range_max))
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
            save_directory, home_directory = '../Figures/', append_time = True,
            casual_format = True)
        gp.classifier.utils.save_all_figures(full_directory)
        shutil.copy2('./%s' % FILENAME , full_directory)

    logging.info('Modeling Done')

    """
    Path Planning
    """

    """ Setup Path Planning """
    xq_now = np.array([[START_POINT1, START_POINT2]])
    horizon = (range_max - range_min) - 1
    n_steps = 30

    if METHOD == 'GREEDY':
        horizon /= n_steps
        n_steps /= n_steps
        METHOD = 'MIE'

    if METHOD == 'RANDOM':
        horizon /= n_steps
        n_steps /= n_steps        

    if METHOD == 'LE':
        theta_bound = np.deg2rad(90)
    else:
        theta_bound = np.deg2rad(180)

    theta_stack_init = -np.deg2rad(10) * np.ones(n_steps)
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
    yq_now = gp.classifier.utils.make_decision(xq_now[[-1]], decision_boundary)

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
    n_trials = N_STEPS
    entropy_linearised_array = np.nan * np.ones(n_trials)
    entropy_linearised_mean_array = np.nan * np.ones(n_trials)
    entropy_true_mean_array = np.nan * np.ones(n_trials)
    entropy_opt_array = np.nan * np.ones(n_trials)
    mistake_ratio_array = np.nan * np.ones(n_trials)
    m_step = 1
    while i_trials < n_trials:

        """ Path Planning """

        # Propose a path
        if m_step <= k_step:
            if METHOD == 'RANDOM':
                xq_abs_opt, theta_stack_opt, entropy_opt = \
                    rh.random_path(theta_stack_init, xq_now[-1], r, 
                        learned_classifier, whitenfn, whitenparams, ranges, 
                        perturb_deg = 60, chaos = CHAOS)
            else:
                xq_abs_opt, theta_stack_opt, entropy_opt = \
                    rh.optimal_path(theta_stack_init, xq_now[-1], r, 
                        learned_classifier, whitenfn, whitenparams, ranges, 
                        theta_stack_low = theta_stack_low, 
                        theta_stack_high = theta_stack_high, 
                        walltime = choice_walltime, xtol_rel = xtol_rel, 
                        ftol_rel = ftol_rel, globalopt = False, 
                        objective = METHOD,
                        n_draws = n_draws_est)
            logging.info('Optimal Joint Entropy: %.5f' % entropy_opt)

            # m_step = rh.correct_lookahead_predictions(xq_abs_opt, 
            # learned_classifier, whitenfn, whitenparams, decision_boundary)
            logging.info('Taking %d steps' % m_step)
        else:
            m_step -= 1
            theta_stack_opt = theta_stack_init.copy()
            xq_abs_opt = rh.forward_path_model(theta_stack_init, r, xq_now[-1])
            logging.info('%d steps left' % m_step)

        # Path steps into the proposed path
        xq_now = xq_abs_opt[:k_step]

        # 
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
        Xw_now, whitenparams = whitenfn(X_now)
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
        xqw_abs_opt = whitenfn(xq_abs_opt, whitenparams)
        xq1_proposed = xq_abs_opt[:, 0][k_step:]
        xq2_proposed = xq_abs_opt[:, 1][k_step:]
        yq_proposed = gp.classifier.classify(gp.classifier.predict(xqw_abs_opt, 
            learned_classifier), y_unique)[k_step:]

        """ Computing Analysis Maps """
        Xqw_plt = whitenfn(Xq_plt, whitenparams)
        Xqw_meas = whitenfn(Xq_meas, whitenparams)
        logging.info('Plot: Caching Predictor...')
        predictor_plt = gp.classifier.query(learned_classifier, Xqw_plt)
        logging.info('Plot: Computing Expectance...')
        exp_plt = gp.classifier.expectance(learned_classifier, predictor_plt)
        logging.info('Plot: Computing Variance...')
        var_plt = gp.classifier.variance(learned_classifier, predictor_plt)

        logging.info('Plot: Computing Linearised Entropy...')
        yq_le_plt = gp.classifier.linearised_entropy(exp_plt, var_plt, 
            learned_classifier)
        logging.info('Plot: Computing Equivalent Standard Deviation...')
        eq_sd_plt = gp.classifier.equivalent_standard_deviation(yq_le_plt)
        logging.info('Plot: Computing Prediction Probabilities...')
        yq_prob_plt = gp.classifier.predict_from_latent(exp_plt, var_plt, 
            learned_classifier, fusemethod = fusemethod)
        logging.info('Plot: Computing Information Entropy...')
        yq_entropy_plt = gp.classifier.entropy(yq_prob_plt)
        logging.info('Plot: Computing Class Predicitons...')
        yq_pred_plt = gp.classifier.classify(yq_prob_plt, y_unique)

        logging.info('Measure: Caching Predictor...')
        predictor_meas = gp.classifier.query(learned_classifier, Xqw_meas)
        logging.info('Measure: Computing Expectance...')
        exp_meas = gp.classifier.expectance(learned_classifier, predictor_meas)
        logging.info('Measure: Computing Covariance...')
        cov_meas = gp.classifier.covariance(learned_classifier, predictor_meas)
        logging.info('Measure: Computing Mistake Ratio...')
        mistake_ratio = (yq_truth_plt - yq_pred_plt).nonzero()[0].shape[0] / \
            yq_truth_plt.shape[0]
        logging.info('Measure: Computing Linearised Joint Entropy...')
        start_time = time.clock()
        entropy_linearised_meas = gp.classifier.linearised_entropy(
            exp_meas, cov_meas, learned_classifier)
        logging.info('Computation took %.4f seconds' % 
            (time.clock() - start_time))
        logging.info('Linearised Joint Entropy: %.4f' % entropy_linearised_meas)
        logging.info('Measure: Computing Average Linearised Entropy...')
        entropy_linearised_mean_meas = yq_le_plt.mean()
        logging.info('Measure: Computing Average Information Entropy...')
        entropy_true_mean_meas = yq_entropy_plt.mean()

        """ Save history """
        mistake_ratio_array[i_trials] = mistake_ratio
        entropy_linearised_array[i_trials] = entropy_linearised_meas
        entropy_linearised_mean_array[i_trials] = entropy_linearised_mean_meas
        entropy_true_mean_array[i_trials] = entropy_true_mean_meas
        entropy_opt_array[i_trials] = entropy_opt
        
        # Find the bounds of the entropy predictions
        if yq_le_plt.max() > 0:
            vmin1 = yq_le_plt.min()
            vmax1 = yq_le_plt.max()
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
        plt.xlim((range_min, range_max))
        plt.ylim((range_min, range_max))

        # Plot linearised entropy
        gp.classifier.utils.visualise_map(yq_le_plt, ranges, 
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

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, 
            head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

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
        plt.xlim((range_min, range_max))
        plt.ylim((range_min, range_max))

        # Plot linearised entropy
        gp.classifier.utils.visualise_map(eq_sd_plt, ranges, 
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

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, 
            head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

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
        plt.xlim((range_min, range_max))
        plt.ylim((range_min, range_max))

        # Plot true entropy
        gp.classifier.utils.visualise_map(yq_entropy_plt, ranges, 
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

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, 
            head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

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
        plt.title('Class predictions [Miss Ratio: %.3f %s]' % 
            (100 * mistake_ratio, '%'), fontsize = fontsize)
        plt.xlabel('x1', fontsize = fontsize)
        plt.ylabel('x2', fontsize = fontsize)
        plt.xlim((range_min, range_max))
        plt.ylim((range_min, range_max))

        # Plot class predictions
        gp.classifier.utils.visualise_map(yq_pred_plt, ranges, 
            boundaries = True, cmap = mycmap2, 
            vmin = y_unique[0], vmax = y_unique[-1])
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

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + 0.2, 0, -0.1, 
            head_width = 0.05, head_length = 0.1, fc = 'w', ec = 'w')

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
        plt.title('Average Marginalised Differential Entropy', 
            fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(5, 1, 4)
        plt.plot(steps_array, entropy_true_mean_array[:(i_trials + 1)])
        plt.title('Average Marginalised Information Entropy', 
            fontsize = fontsize)
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

    """ Save Final Results """
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
        except Exception:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()

    # Show everything!
    plt.show()

if __name__ == "__main__":
    main()