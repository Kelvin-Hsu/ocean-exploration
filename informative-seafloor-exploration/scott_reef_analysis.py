"""
Module  : Ocean Environment Modeling
File    : main.py

Author  : Kelvin Hsu
Date    : 1st July 2015
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from computers import gp
import computers.unsupervised.whitening as pre
import time
import logging
import sys
import os
import sea
import shutil

def kerneldef(h, k):
    """Define the kernel used in the classifier"""
    return  h(1e-3, 1e3, 10)*k('gaussian', 
            [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), 
            h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])

def main():

    """ Test Options """
    FILENAME = 'scott_reef_analysis.py'
    T_SEED = sea.io.parse('-tseed', 250)
    Q_SEED = sea.io.parse('-qseed', 500)
    NOTRAIN = sea.io.parse('-skiptrain', False)
    N_TRAIN = sea.io.parse('-ntrain', 200)
    N_QUERY = sea.io.parse('-nquery', 100000)

    METHOD = sea.io.parse('-method', 'LDE')
    GREEDY = sea.io.parse('-greedy', False)
    N_TRIALS = sea.io.parse('-ntrials', 200)
    START_POINT1 = sea.io.parse('-start', 375000.0, arg = 1)
    START_POINT2 = sea.io.parse('-start', 8440000.0, arg = 2)
    H_STEPS = sea.io.parse('-hsteps', 30)
    HORIZON = sea.io.parse('-horizon', 5000.0)
    CHAOS = sea.io.parse('-chaos', False)
    M_STEP = sea.io.parse('-mstep', 1)
    N_DRAWS = sea.io.parse('-ndraws', 5000)
    
    # NOTRAIN = True
    """Model Options"""
    SAVE_RESULTS = True

    approxmethod = 'laplace'
    multimethod = 'OVA'
    fusemethod = 'EXCLUSION'
    responsename = 'probit'
    batchstart = True
    batchlearn = False
    walltime = 3600.0
    train = not NOTRAIN
    white_fn = pre.standardise

    n_train = N_TRAIN
    n_query = N_QUERY

    """Visualisation Options"""
    mycmap = cm.get_cmap(name = 'jet', lut = None)
    vis_fix_range = True
    vis_x_min = 360000
    vis_x_max = 390000
    vis_y_min = 8430000
    vis_y_max = 8450000
    vis_range = (vis_x_min, vis_x_max, vis_y_min, vis_y_max)
    colorcenter_analysis = 'mean'
    colorcenter_lde = 'mean'
    y_names_all = [ 'None',
                    'Under-Exposed', 
                    'Under-Exposed',
                    'Barron Sand 1',
                    'Low Density Coral 1',
                    'Sand Biota 1',
                    'Low Density Coral 2',
                    'Dense Coral 1',
                    'Dense Coral 2',
                    'Dense Coral 3',
                    'Sand Biota 2',
                    'Low Density Coral 3',
                    'Low Density Coral 4',
                    'Patch 1',
                    'Patch 2',
                    'Patch 3',
                    'Barron Sand 2',
                    'Sand Biota 3',
                    'Over-Exposed',
                    'Barron Sand 3',
                    'Under-Exposed',
                    'Under-Exposed',
                    'Sand Biota 4',
                    'Misc',
                    'Under-Exposed']

    assert len(y_names_all) == 25
               
    """Initialise Result Logging"""
    if SAVE_RESULTS:
        home_directory = "../../../Results/scott-reef/"
        save_directory = "t%d_q%d_ts%d_qs%d_method_%s%s_start%.1f%.1f_"\
        "hsteps%d_horizon%.1f/" % (N_TRAIN, N_QUERY, T_SEED, Q_SEED, 
                METHOD, '_GREEDY' if GREEDY else '', 
                START_POINT1, START_POINT2, H_STEPS, HORIZON)
        full_directory = gp.classifier.utils.create_directories(
            save_directory, 
            home_directory = home_directory, append_time = True)
        textfilename = '%slog.txt' % full_directory

    """Logging Options"""
    logging.basicConfig(level = logging.DEBUG,
                        format =    '%(asctime)s %(name)-12s '\
                                    '%(levelname)-8s %(message)s',
                        datefmt = '%m-%d %H:%M',
                        filename = textfilename,
                        filemode = 'a')
    gp.classifier.set_multiclass_logging_level(logging.DEBUG)

    # Define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # Set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    # Tell the handler to use this format
    console.setFormatter(formatter)

    # Add the handler to the root logger
    logging.getLogger().addHandler(console)

    """Process Options"""
    test_options  = {   'T_SEED': T_SEED,
                        'Q_SEED': Q_SEED,
                        'N_TRAIN': N_TRAIN,
                        'N_QUERY': N_QUERY,
                        'METHOD': METHOD,
                        'GREEDY': GREEDY,
                        'N_TRIALS': N_TRIALS,
                        'START_POINT1': START_POINT1,
                        'START_POINT2': START_POINT2,
                        'H_STEPS': H_STEPS,
                        'HORIZON': HORIZON,
                        'CHAOS': CHAOS,
                        'M_STEP': M_STEP}
    model_options = {   'approxmethod': approxmethod,
                        'multimethod': multimethod,
                        'fusemethod': fusemethod,
                        'responsename': responsename,
                        'batchstart': batchstart,
                        'batchlearn': batchlearn,
                        'walltime': walltime,
                        'train': train}

    logging.info(sys.argv)
    logging.info(test_options)
    logging.info(model_options)

    """File Locations"""
    directory_data = '../../../Data/'
    filename_training_data = 'training_data_unmerged.npz'
    filename_query_points = 'query_points.npz'
    filename_truth = directory_data + 'truthmodel_t800_q100000_ts250_qs500.npz'
    filename_start = directory_data + 'finalmodel_t200_q100000_ts250_qs500'\
        '_method_LDE_start377500_8440000_hsteps30_horizon5000.npz'

    """Sample Training Data and Query Points"""
    X, F, y, Xq, Fq, i_train, i_query = \
        sea.io.sample(*sea.io.load(directory_data, 
            filename_training_data, filename_query_points), 
            n_train = n_train, n_query = n_query,
            t_seed = T_SEED, q_seed = Q_SEED)

    yq_truth = sea.io.load_ground_truth(filename_truth, 
        assert_query_seed = Q_SEED)

    y_unique = np.unique(y)
    assert y_unique.shape[0] == 17
    logging.info('There are %d unique labels' % y_unique.shape[0])

    y_names = [y_names_all[i] for i in y_unique.astype(int)]
    logging.info('Habitat Labels: {0}'.format(y_names))

    """Whiten the feature space"""
    logging.info('Applying whitening on training and query features...')
    feature_fn = sea.feature.compose(Xq, Fq, white_fn)
    Fw, white_params = white_fn(F)
    Fqw = white_fn(Fq, params = white_params)

    k_features = F.shape[1]
    assert k_features == 5
    feature_names = [   'Bathymetry (Depth)', 
                        'Aspect (Short Scale)',
                        'Rugosity (Short Scale)',
                        'Aspect (Long Scale)',
                        'Rugosity (Long Scale)']
    logging.info('Whitening Parameters:')
    logging.info(white_params)

    """Visualise Sampled Training Locations"""
    fig = plt.figure(figsize = (19.2, 10.8))
    plt.scatter(
        X[:, 0], X[:, 1], 
        marker = 'x', c = y, 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)
    sea.vis.describe_plot(title = 'Training Labels', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
        vis_range = vis_range, aspect_equal = True)

    """Visualise Features at Sampled Query Locations"""
    for k in range(k_features):
        fig = plt.figure(figsize = (19.2, 10.8))
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            marker = 'x', c = Fq[:, k], s = 5, 
            cmap = mycmap, colorcenter = 'mean')
        sea.vis.describe_plot(clabel = '%s (Raw)' % feature_names[k])
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            marker = 'x', c = Fqw[:, k], s = 5,
            cmap = mycmap, colorcenter = 'mean')
        sea.vis.describe_plot(
            title = 'Feature: %s at Query Points' % feature_names[k], 
            xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
            clabel = '%s (Whitened)' % feature_names[k],
            vis_range = vis_range, aspect_equal = True)

    """Classifier Training"""
    logging.info('===Begin Classifier Training===')
    logging.info('Number of training points: %d' % n_train)

    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # User can choose to batch start each binary classifier with different
    # initial hyperparameters for faster training
    if batchstart:
        # TSEED 250 NTRAIN 300
        initial_hyperparams = \
                                [    [320.57877936128614, 1.1585314538133855, 16.971830202892182, 13.2982146397511, 3.5903175970474051, 157.39424107768139], \
                                     [94.92405400574269, 1.2788122462876559, 66.128048600437225, 5.8584634929906532, 2.3376269482977525, 12.938852433246538], \
                                     [7.2316752676281784, 211.01282813515252, 605.21124029888824, 118.95015047046942, 36.415548937170669, 269.17747017168222], \
                                     [7.3035958524895257, 5.2818319863811576, 369.95771303726838, 36.337681287162432, 56.867452183483685, 11.468838185312663], \
                                     [2.7046818132602435, 38.770906017912445, 16.142889638319531, 19.017568262215736, 108.35872235764178, 12.752297332848453], \
                                     [5.1665202238361783, 2.508749221008296, 57.15411954328259, 10.26672628635392, 53.461595052564192, 34.385700775257703], \
                                     [2.0748360321398738, 1.0019456339096766, 46.049013888326435, 21.594510373841985, 12.090471875156618, 26.23829355131695], \
                                     [3.9424935896691746, 1.2770305891532705, 220.0814074951281, 63.931498176312367, 67.241596768503825, 17.990743966922299], \
                                     [5.1316363440138382, 2.4803332422329905, 233.1152982720449, 119.50661835434632, 57.633584194008264, 139.68353543314493], \
                                     [31.362876683396554, 2.2271043809429054, 46.110959686547844, 52.1283686477149, 2.4091611534708366, 92.121824783697875], \
                                     [2.8521213156866261, 4.3591470146739466, 76.498779011353079, 38.284082633970044, 26.878550897240991, 2.6681145719573389], \
                                     [5.3226281295231752, 5.5118656576536669, 86.291450037433933, 228.53831221134874, 56.459752612169567, 18.718760980202806], \
                                     [4.3985822329997442, 1.2643285860275568, 193.0207848771841, 172.00274323050451, 155.19849940832842, 115.96329482769362], \
                                     [17.965761637190241, 1.9304086197451131, 50.180636402767199, 149.50394158494592, 12.061707174982143, 31.765932106221591], \
                                     [57.4423127300958, 2.0036456868482015, 9.7157285648070051, 94.011010785775269, 5.8328137782124951, 23.874860922848949], \
                                     [2.7124818082845481, 4.4512881809223028, 43.89719061954132, 114.15707647584941, 184.06607505787818, 2.4421440364983495], \
                                     [7.587190958981215, 162.42170269550411, 164.35681334098788, 303.17711599826504, 65.201091109094804, 531.88153321387961] ]
        # TSEED 250 NTRAIN 100
        initial_hyperparams = \
                                [    [158.04660629989138, 1.2683731889725351, 66.115010215389162, 49.467620758110257, 2.1003587537731203, 148.19413243571267], \
                                     [65.998503029331715, 0.96182325466796015, 76.537506649529078, 8.7072874430795792, 2.7122005803599105, 31.811834175256053], \
                                     [7.2059309204166064, 248.71972897462479, 640.00239325817472, 168.88943619928102, 90.693076836996767, 262.04071654280534], \
                                     [6.0223358705627561, 5.7983854605769629, 7.151322371121573, 98.235863530785863, 108.97929055450719, 3.4049522081768151], \
                                     [2.8661028631425438, 202.71822125606951, 71.427958520531689, 67.841056622412466, 396.4966008975731, 8.7928316190417863], \
                                     [4.2274366302519937, 2.6787248415957619, 155.52642217518203, 12.902809348832989, 78.595731986539263, 91.408564485980548], \
                                     [1.9655515448696013, 1.0078459070165113, 47.264505099891736, 90.096628012215518, 39.449679781547545, 12.422258217851045], \
                                     [3.1222954891045123, 2.4207646445523401, 238.89962477023275, 110.82548792960249, 96.535214824647056, 13.310640119621617], \
                                     [4.9184407605671376, 2.2582428037913012, 220.5347539837725, 98.366730030395019, 51.082771850913375, 29.11755699767917], \
                                     [46.140309605544743, 3.4122114646182515, 66.780957403339869, 56.618087864345419, 3.4707008796589727, 8.5854627686410119], \
                                     [4.3654723196710217, 4.8059179997026096, 190.44003623250873, 122.8813398402322, 95.879994772087869, 3.3765033232957848], \
                                     [4.4788217098193668, 10.934823441369865, 123.33728936692876, 382.04572230065276, 227.51098327130356, 79.977071466413577], \
                                     [4.2294957554111283, 1.3726514772357896, 208.19938841666934, 174.7937853783985, 10.037269529873239, 127.83511529606474], \
                                     [27.075374606311115, 1.814069988416549, 7.7469247716502938, 170.97327917010006, 40.037475847074973, 56.043993677737269], \
                                     [83.400751642781657, 1.6747947434483126, 17.596540717504126, 112.41698370483411, 4.6137167168846629, 42.078275936624557], \
                                     [3.0901973040826474, 12.319455336644344, 70.719245134201159, 140.9221718578234, 224.77657336458043, 4.57995836705937], \
                                     [7.0310415618402358, 211.04475504456954, 202.37890056818438, 261.95210773212585, 156.94082106951919, 511.27885698486267] ]

        batch_config = gp.batch_start(optimiser_config, initial_hyperparams)
        logging.info('Using Batch Start Configuration')

    else:
        batch_config = optimiser_config
    
    # Obtain the response function
    responsefunction = gp.classifier.responses.get(responsename)

    # Train the classifier!
    logging.info('Learning...')
    if batchlearn:
        previous_history = np.load(filename_start)
        learned_classifier = list(previous_history['learned_classifier'])
        white_params = previous_history['white_params']
        batch_config = \
            gp.classifier.batch_start(optimiser_config, learned_classifier)
        Fqw = white_fn(Fq, white_params)
    else:
        learned_classifier = gp.classifier.learn(Fw, y, 
            kerneldef, responsefunction, batch_config, 
            multimethod = multimethod, approxmethod = approxmethod, 
            train = train, ftol = 1e-10)

    # Print the learnt kernel with its hyperparameters
    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(print_function, 
        learned_classifier, y_unique)

    # Print the matrix of learned classifier hyperparameters
    logging.info('Matrix of learned hyperparameters')
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)

    """Classifier Prediction"""

    yq_pred, yq_mie, yq_lde = sea.model.predictions(learned_classifier, Fqw,
        fusemethod = fusemethod)
    yq_esd = gp.classifier.equivalent_standard_deviation(yq_lde)
    miss_ratio = sea.model.miss_ratio(yq_pred, yq_truth)
    yq_lde_mean = yq_lde.mean()
    yq_mie_mean = yq_mie.mean()
    logging.info('Miss Ratio: {0:.2f}%'.format(100 * miss_ratio))
    logging.info('Average Marginalised Linearised Differential Entropy: '\
        '{0:.2f}'.format(yq_lde_mean))
    logging.info('Average Marginalised Information Entropy: '\
        '{0:.2f}'.format(yq_mie_mean))

    """Visualise Query Prediction and Entropy"""
    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_truth, s = 5, 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)
    sea.vis.describe_plot(title = 'Synthetic Ground Truth', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_pred, s = 5, 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)
    sea.vis.describe_plot(
        title = 'Query Predictions [Miss Ratio: {0:.2f}%]'.format(
                100 * miss_ratio), 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_mie, s = 5, cmap = cm.coolwarm, 
        colorcenter = colorcenter_analysis)
    sea.vis.describe_plot(title = 'Query Information Entropy', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Information Entropy',
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = np.log(yq_mie), s = 5, cmap = cm.coolwarm, 
        colorcenter = colorcenter_analysis)
    sea.vis.describe_plot(title = 'Log Query Information Entropy', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Information Entropy',
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_lde, s = 5, cmap = cm.coolwarm, 
        colorcenter = colorcenter_lde)
    sea.vis.describe_plot(title = 'Query Linearised Differential Entropy', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Differential Entropy',
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_esd, s = 5, cmap = cm.coolwarm, 
        colorcenter = colorcenter_analysis)
    sea.vis.describe_plot(title = 'Query Equivalent Standard Deviation', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Standard Deviation',
        vis_range = vis_range, aspect_equal = True)

    """Visualise Query Draws"""

    if SAVE_RESULTS:
        gp.classifier.utils.save_all_figures(full_directory)
        shutil.copy2('./%s' % FILENAME , full_directory)
        np.savez('%sinitialmodel.npz' % full_directory, 
                learned_classifier = learned_classifier,
                t_seed = T_SEED, q_seed = Q_SEED,
                n_train = n_train, n_query = n_query,
                i_train = i_train, i_query = i_query,
                yq_pred = yq_pred, yq_mie = yq_mie, yq_lde = yq_lde,
                white_params = white_params)

    """Informative Seafloor Exploration: Setup"""
    xq_now = np.array([[START_POINT1, START_POINT2]])
    xq_now = feature_fn.closest_locations(xq_now)
    horizon = HORIZON
    h_steps = H_STEPS

    if GREEDY or (METHOD == 'RANDOM') or (METHOD == 'FIXED'):
        horizon /= h_steps
        h_steps /= h_steps 

    if METHOD == 'LDE':
        theta_bound = np.deg2rad(20)
        theta_bounds = np.linspace(theta_bound, np.deg2rad(60), num = h_steps) 
        theta_stack_low  = -theta_bounds
        theta_stack_high = +theta_bounds
        xtol_rel = 1e-2
        ftol_rel = 1e-3
    else:
        theta_bound = np.deg2rad(270)
        theta_bounds = theta_bound * np.ones(h_steps)
        theta_stack_low  = -theta_bounds
        theta_stack_high = +theta_bounds
        xtol_rel = 1e-1
        ftol_rel = 1e-1
    ctol = 1e-10

    theta_stack_init = np.deg2rad(0) * np.ones(h_steps)
    theta_stack_init[0] = np.deg2rad(180)
    theta_stack_low[0] = 0.0
    theta_stack_high[0] = 2 * np.pi
    r = horizon/h_steps
    choice_walltime = 1500.0

    k_step = 1
    m_step = 1

    bound = 100

    """Informative Seafloor Exploration: Initialisation"""
    # The observed data till now
    X_now = X.copy()
    y_now = y.copy()

    # Observe the current location
    i_observe = sea.feature.closest_indices(xq_now, Xq)
    yq_now = yq_truth[i_observe]

    # Add the observed data to the training set
    X_now = np.concatenate((X_now, xq_now[[-1]]), axis = 0)
    y_now = np.append(y_now, yq_now)

    # Add the new location to the array of travelled coordinates
    xq1_nows = xq_now[:, 0]
    xq2_nows = xq_now[:, 1]
    yq_nows = yq_now.copy()

    # Plot the current situation
    fig1 = plt.figure(figsize = (19.2, 10.8))
    fig2 = plt.figure(figsize = (19.2, 10.8))
    fig3 = plt.figure(figsize = (19.2, 10.8))
    fig4 = plt.figure(figsize = (19.2, 10.8))
    fig5 = plt.figure(figsize = (19.2, 10.8))

    # Start exploring
    i_trials = 0
    n_trials = N_TRIALS
    miss_ratio_array = np.nan * np.ones(n_trials)
    yq_mie_mean_array = np.nan * np.ones(n_trials)
    yq_lde_mean_array = np.nan * np.ones(n_trials)
    entropy_opt_array = np.nan * np.ones(n_trials)
    yq_esd_mean_array = np.nan * np.ones(n_trials)

    if (METHOD == 'FIXED') or (METHOD == 'LDE'):
        turns = np.zeros(n_trials)
        turns[[49, 99, 149]] = np.deg2rad(-90.0)

        # turns = np.linspace(np.deg2rad(20), np.deg2rad(0), num = n_trials)
        # turns = np.deg2rad(30) * np.sin(np.linspace(0, 20*np.pi, num = n_trials))
        # turns = np.linspace(np.deg2rad(60), np.deg2rad(0), num = n_trials)

    while i_trials < n_trials:

        if METHOD == 'LDE':
            theta_stack_init[0] += turns[i_trials]
            theta_stack_init[0] = np.mod(theta_stack_init[0], 2 * np.pi)

        # Propose a path
        if m_step <= k_step:
            if METHOD == 'RANDOM':
                xq_path, theta_stack_opt, entropy_opt = \
                    sea.explore.random_path(theta_stack_init, r, xq_now[-1], 
                        learned_classifier, feature_fn, white_params, 
                        bound = bound, 
                        chaos = CHAOS)
            elif METHOD == 'FIXED':
                xq_path, theta_stack_opt, entropy_opt = \
                    sea.explore.fixed_path(theta_stack_init, r, xq_now[-1], 
                        learned_classifier, feature_fn, white_params,
                        bound = bound, 
                        current_step = i_trials, 
                        turns = turns)
            else:
                xq_path, theta_stack_opt, entropy_opt = \
                    sea.explore.optimal_path(theta_stack_init, r, xq_now[-1],
                        learned_classifier, feature_fn, white_params,
                        objective = METHOD,
                        turn_limit = theta_bound,
                        bound = bound,
                        theta_stack_low = theta_stack_low,
                        theta_stack_high = theta_stack_high,
                        walltime = choice_walltime,
                        xtol_rel = xtol_rel,
                        ftol_rel = ftol_rel,
                        ctol = ctol,
                        globalopt = False,
                        n_draws = N_DRAWS)
            logging.info('Optimal Joint Entropy: %.5f' % entropy_opt)

            m_step = M_STEP
            logging.info('Taking %d steps' % m_step)
        else:
            m_step -= 1
            theta_stack_opt = theta_stack_init.copy()
            xq_path = sea.explore.forward_path_model(theta_stack_init, 
                r, xq_now[-1])
            logging.info('%d steps left' % m_step)

        # Path steps into the proposed path
        xq_now = xq_path[:k_step]

        # Initialise the next path angles
        theta_stack_init = sea.explore.shift_path(theta_stack_opt, 
            k_step = k_step, theta_bounds = theta_bounds)
        np.clip(theta_stack_init, 
            theta_stack_low + 1e-4, theta_stack_high - 1e-4, 
            out = theta_stack_init)

        # Observe the current location
        i_observe = sea.feature.closest_indices(xq_now, Xq)
        yq_now = yq_truth[i_observe]

        # Add the observed data to the training set
        X_now = np.concatenate((X_now, xq_now), axis = 0)
        y_now = np.append(y_now, yq_now)

        # Add the new location to the array of travelled coordinates
        xq1_nows = np.append(xq1_nows, xq_now[:, 0])
        xq2_nows = np.append(xq2_nows, xq_now[:, 1])
        yq_nows = np.append(yq_nows, yq_now)

        # Update that into the model
        Fw_now, white_params = feature_fn(X_now)
        logging.info('Learning Classifier...')
        batch_config = \
            gp.classifier.batch_start(optimiser_config, learned_classifier)
        try:
            learned_classifier = gp.classifier.learn(Fw_now, y_now, 
                kerneldef, responsefunction, batch_config, 
                multimethod = multimethod, approxmethod = approxmethod,
                train = True, ftol = 1e-6)
        except Exception as e:
            logging.warning('Training failed: {0}'.format(e))
            try:
                learned_classifier = gp.classifier.learn(Fw_now, y_now, 
                    kerneldef, responsefunction, batch_config, 
                    multimethod = multimethod, approxmethod = approxmethod,
                    train = False, ftol = 1e-6)
            except Exception as e:
                logging.warning('Learning also failed: {0}'.format(e))
                pass    
        logging.info('Finished Learning')

        # This is the finite horizon optimal route
        fqw_opt = feature_fn(xq_path, white_params)
        xq1_path = xq_path[:, 0][k_step:]
        xq2_path = xq_path[:, 1][k_step:]
        yq_opt = gp.classifier.classify(gp.classifier.predict(fqw_opt, 
            learned_classifier), y_unique)[k_step:]

        """ Computing Analysis Maps """
        Fqw = white_fn(Fq, white_params)

        yq_pred, yq_mie, yq_lde = \
            sea.model.predictions(learned_classifier, Fqw, 
                fusemethod = fusemethod)
        yq_esd = gp.classifier.equivalent_standard_deviation(yq_lde)
        miss_ratio = sea.model.miss_ratio(yq_pred, yq_truth)
        yq_mie_mean = yq_mie.mean()
        yq_lde_mean = yq_lde.mean()
        yq_esd_mean = yq_esd.mean()
        logging.info('Miss Ratio: {0:.2f}%'.format(100 * miss_ratio))
        logging.info('Average Marginalised Linearised Differential Entropy: '\
            '{0:.2f}'.format(yq_lde_mean))
        logging.info('Average Marginalised Information Entropy: '\
            '{0:.2f}'.format(yq_mie_mean))


        """ Save history """
        miss_ratio_array[i_trials] = miss_ratio
        yq_mie_mean_array[i_trials] = yq_mie_mean
        yq_lde_mean_array[i_trials] = yq_lde_mean
        yq_esd_mean_array[i_trials] = yq_esd_mean
        entropy_opt_array[i_trials] = entropy_opt

        # Find the bounds of the entropy predictions
        vmin1 = yq_lde.min()
        vmax1 = yq_lde.max()
        vmin2 = yq_mie.min()
        vmax2 = yq_mie.max()
        vmin3 = yq_esd.min()
        vmax3 = yq_esd.max()

        logging.info('Plotting...')

        """ Linearised Entropy Map """

        # Prepare Figure 1
        plt.figure(fig1.number)
        plt.clf()
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            marker = 'x', c = yq_lde, s = 5, 
            cmap = cm.coolwarm, colorcenter = colorcenter_lde)
        sea.vis.describe_plot(title = 'Query Linearised Differential Entropy', 
            xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
            clabel = 'Differential Entropy',
            vis_range = vis_range, aspect_equal = True)

        # Plot the path on top
        sea.vis.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none', 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)
        sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
        sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            linewidth = 2, marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + r, 0, -r/4, 
            head_width = r/4, head_length = r/4, fc = 'k', ec = 'k')

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%slde_step%d.png' 
            % (full_directory, i_trials + 1))

        # Plot the proposed path
        sea.vis.scatter(xq1_path, xq2_path, c = yq_opt, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        sea.vis.plot(xq1_path, xq2_path, c = 'k', linewidth = 2)

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%slde_propose_step%d.png' 
            % (full_directory, i_trials + 1))

        """ Equivalent Standard Deviation Map """

        # Prepare Figure 2
        plt.figure(fig2.number)
        plt.clf()
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            marker = 'x', c = yq_esd, s = 5, 
            cmap = cm.coolwarm, colorcenter = colorcenter_analysis)
        sea.vis.describe_plot(title = 'Query Equivalent Standard Deviation', 
            xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
            clabel = 'Standard Deviation',
            vis_range = vis_range, aspect_equal = True)

        # Plot the path on top
        sea.vis.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none', 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)
        sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
        sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            linewidth = 2, marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + r, 0, -r/4, 
            head_width = r/4, head_length = r/4, fc = 'k', ec = 'k')

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sesd_step%d.png' 
            % (full_directory, i_trials + 1))

        # Plot the proposed path
        sea.vis.scatter(xq1_path, xq2_path, c = yq_opt, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        sea.vis.plot(xq1_path, xq2_path, c = 'k', linewidth = 2)

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%sesd_propose_step%d.png' 
            % (full_directory, i_trials + 1))

        """ True Entropy Map """

        # Prepare Figure 3
        plt.figure(fig3.number)
        plt.clf()
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            marker = 'x', c = yq_mie, s = 5, 
            cmap = cm.coolwarm, colorcenter = colorcenter_analysis)
        sea.vis.describe_plot(title = 'Query Prediction Information Entropy', 
            xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
            clabel = 'Information Entropy',
            vis_range = vis_range, aspect_equal = True)

        # Plot the path on top
        sea.vis.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none', 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)
        sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
        sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%smie_step%d.png' 
            % (full_directory, i_trials + 1))

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            linewidth = 2, marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + r, 0, -r/4, 
            head_width = r/4, head_length = r/4, fc = 'k', ec = 'k')

        # Plot the proposed path
        sea.vis.scatter(xq1_path, xq2_path, c = yq_opt, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        sea.vis.plot(xq1_path, xq2_path, c = 'k', linewidth = 2)

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%smie_propose_step%d.png' 
            % (full_directory, i_trials + 1))

        """ Class Prediction Map """

        # Prepare Figure 4
        plt.figure(fig4.number)
        plt.clf()
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            marker = 'x', c = yq_pred, s = 5, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)
        sea.vis.describe_plot(
            title = 'Query Predictions [Miss Ratio: {0:.2f}%]'.format(
                100 * miss_ratio), 
            xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
            clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
            vis_range = vis_range, aspect_equal = True)

        # Plot the path on top
        sea.vis.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
            facecolors = 'none', 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)
        sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
        sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Plot the horizon
        gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
            linewidth = 2, marker = '.')

        plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + r, 0, -r/4, 
            head_width = r/4, head_length = r/4, fc = 'k', ec = 'k')

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%spred_step%d.png' 
            % (full_directory, i_trials + 1))

        # Plot the proposed path
        sea.vis.scatter(xq1_path, xq2_path, c = yq_opt, 
            s = 60, marker = 'D', 
            vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
        sea.vis.plot(xq1_path, xq2_path, c = 'k', linewidth = 2)

        # Save the plot
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%spred_propose_step%d.png' 
            % (full_directory, i_trials + 1))


        # Prepare Figure 5
        plt.figure(fig5.number)
        plt.clf()
        fontsize = 24
        ticksize = 14

        steps_array = np.arange(i_trials + 1) + 1
        ax = plt.subplot(4, 1, 1)
        plt.plot(steps_array, 100 * miss_ratio_array[:(i_trials + 1)])
        plt.title('Percentage of Prediction Misses', fontsize = fontsize)
        plt.ylabel('Misses (%)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(4, 1, 2)
        plt.plot(steps_array, yq_lde_mean_array[:(i_trials + 1)])
        plt.title('Average Marginalised Differential Entropy', 
            fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(4, 1, 3)
        plt.plot(steps_array, yq_mie_mean_array[:(i_trials + 1)])
        plt.title('Average Marginalised Information Entropy', 
            fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(4, 1, 4)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.plot(steps_array, entropy_opt_array[:(i_trials + 1)])
        plt.title('Entropy Metric of Proposed Path', fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)

        plt.xlabel('Steps', fontsize = fontsize)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize) 
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize) 

        # Save the plot
        plt.tight_layout()
        plt.savefig('%shistory%d.png' 
            % (full_directory, i_trials + 1))
        logging.info('Plotted and Saved Iteration')

        # Move on to the next step
        i_trials += 1
    
        np.savez('%shistory.npz' % full_directory, 
            learned_classifier = learned_classifier,
            miss_ratio_array = miss_ratio_array,
            yq_lde_mean_array = yq_lde_mean_array,
            yq_mie_mean_array = yq_mie_mean_array,
            entropy_opt_array = entropy_opt_array,
            yq_esd_mean_array = yq_esd_mean_array,
            yq_lde = yq_lde,
            yq_mie = yq_mie,
            yq_pred = yq_pred,
            white_params = white_params,
            X_now = X_now,
            Fw_now = Fw_now,
            y_now = y_now,
            xq1_nows = xq1_nows,
            xq2_nows = xq2_nows,
            yq_nows = yq_nows)
        logging.info('White Params: {0}'.format(white_params))

    np.savez('%shistory.npz' % full_directory, 
        learned_classifier = learned_classifier,
        miss_ratio_array = miss_ratio_array,
        yq_lde_mean_array = yq_lde_mean_array,
        yq_mie_mean_array = yq_mie_mean_array,
        entropy_opt_array = entropy_opt_array,
        yq_esd_mean_array = yq_esd_mean_array,
        yq_lde = yq_lde,
        yq_mie = yq_mie,
        yq_pred = yq_pred,
        white_params = white_params,
        X_now = X_now,
        Fw_now = Fw_now,
        y_now = y_now,
        xq1_nows = xq1_nows,
        xq2_nows = xq2_nows,
        yq_nows = yq_nows)

    plt.show()

if __name__ == "__main__":
    main()
