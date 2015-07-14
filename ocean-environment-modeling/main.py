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

def kerneldef(h, k):
    """Define the kernel used in the classifier"""
    return  h(1e-3, 1e3, 10)*k('gaussian', 
            [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), 
            h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])

def main():

    """Model Options"""
    SAVE_RESULTS = True

    approxmethod = 'laplace' # 'laplace' or 'pls'
    multimethod = 'AVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
    responsename = 'probit' # 'probit' or 'logistic'
    batchstart = False
    walltime = 10*3600.0
    train = True

    n_train_sample = 1000
    n_query_sample = 2000
    n_draws = 1
    rows_subplot = 1
    cols_subplot = 1

    assert rows_subplot * cols_subplot >= n_draws

    generate_draw = True
    
    """Visualisation Options"""
    mycmap = cm.jet
    vis_fix_range = True
    vis_x_min = 365000
    vis_x_max = 390000
    vis_y_min = 8430000
    vis_y_max = 8448000

    """Initialise Result Logging"""
    if SAVE_RESULTS:

    	#"C:/Users/kkeke_000/Dropbox/Thesis/" \
        home_directory = "../../../Results/ocean-exploration/"

        save_directory = "scott_reef__training_%d_query_%d/" % \
        				(n_train_sample, n_query_sample)

        full_directory = gp.classifier.utils.create_directories(save_directory, 
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
    model_options = {   'trainingsample': n_train_sample,
    					'querysample': n_query_sample,
    					'approxmethod': approxmethod,
                        'multimethod': multimethod,
                        'fusemethod': fusemethod,
                        'responsename': responsename,
                        'batchstart': batchstart,
                        'walltime': walltime,
                        'train': train}

    logging.info(model_options)

    """File Locations"""
    directory_data = '../../../Data/'
    filename_training_data = 'training_data_unmerged.npz'
    filename_query_points = 'query_points.npz'

    """Load Data"""
    (training_locations, training_features, training_labels, \
        query_locations, query_features) = \
            load(directory_data, filename_training_data, filename_query_points)

    n_train = training_features.shape[0]
    n_query = query_features.shape[0]
    k_features = training_features.shape[1]
    feature_names = [   'Bathymetry (Depth)', 
                        'Aspect (Short Scale)',
                        'Rugosity (Short Scale)',
                        'Aspect (Long Scale)',
                        'Rugosity (Long Scale)']

    logging.info('Raw Number of Training Points: %d' % n_train)
    logging.info('Raw Number of Query Points: %d' % n_query)

    """Sample Training Data and Query Points"""
    (training_locations_sample, training_features_sample, 
    training_labels_sample, 
    query_locations_sample, query_features_sample) \
        = sample(training_locations, training_features, training_labels, 
        query_locations, query_features, 
        n_train_sample = n_train_sample, n_query_sample = n_query_sample)

    unique_labels_sample = np.unique(training_labels_sample)

    """Whiten the feature space"""
    logging.info('Applying whitening on training and query features...')
    (training_features_sample_whiten, whiten_params) = \
        pre.standardise(training_features_sample)

    query_features_sample_whiten = \
        pre.standardise(query_features_sample, params = whiten_params)

    logging.info('Whitening Parameters:')
    logging.info(whiten_params)

    """Visualise Sampled Training Locations"""
    fig = plt.figure(figsize = (19.2, 10.8))
    plt.scatter(
        training_locations_sample[:, 0], training_locations_sample[:, 1], 
        marker = 'x', c = training_labels_sample, 
        vmin = unique_labels_sample[0], vmax = unique_labels_sample[-1], 
        cmap = mycmap)
    plt.title('Training Labels')
    plt.xlabel('x [Eastings (m)]')
    plt.ylabel('y [Northings (m)]')

    cbar = plt.colorbar()
    cbar.set_ticks(unique_labels_sample)
    cbar.set_ticklabels(unique_labels_sample)
    cbar.set_label('Habitat Labels')
    if vis_fix_range:
        plt.xlim((vis_x_min, vis_x_max))
        plt.ylim((vis_y_min, vis_y_max))
    plt.gca().set_aspect('equal', adjustable = 'box')

    """Visualise Features at Sampled Query Locations"""
    for k in range(k_features):
        fig = plt.figure(figsize = (19.2, 10.8))
        plt.scatter(
            query_locations_sample[:, 0], query_locations_sample[:, 1], 
            marker = 'x', c = query_features_sample[:, k], cmap = mycmap)
        cbar1 = plt.colorbar()
        cbar1.set_label('%s (Raw)' % feature_names[k])
        plt.scatter(
            query_locations_sample[:, 0], query_locations_sample[:, 1], 
            marker = 'x', c = query_features_sample_whiten[:, k], 
            cmap = mycmap)
        cbar2 = plt.colorbar()
        cbar2.set_label('%s (Whitened)' % feature_names[k])
        plt.title('Feature: %s at Query Points' % feature_names[k])
        plt.xlabel('x [Eastings (m)]')
        plt.ylabel('y [Northings (m)]')

        if vis_fix_range:
            plt.xlim((vis_x_min, vis_x_max))
            plt.ylim((vis_y_min, vis_y_max))
        plt.gca().set_aspect('equal', adjustable = 'box')

        logging.info('Feature %d' % k)
        logging.info('\tPre-Whiten')
        logging.info('\t\tTraining: [%.4f, %.4f]' % \
            (training_features_sample[:, k].min(), 
            training_features_sample[:, k].max()))
        logging.info('\t\tQuery: [%.4f, %.4f]' % \
            (query_features_sample[:, k].min(), 
            query_features_sample[:, k].max()))

        logging.info('\tWhiten')
        logging.info('\t\tTraining: [%.4f, %.4f]' % \
            (training_features_sample_whiten[:, k].min(), 
            training_features_sample_whiten[:, k].max()))
        logging.info('\t\tQuery: [%.4f, %.4f]' % \
            (query_features_sample_whiten[:, k].min(), 
            query_features_sample_whiten[:, k].max()))

    """Classifier Training"""
    logging.info('===Begin Classifier Training===')
    logging.info('Number of training points: %d' % n_train_sample)

    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # User can choose to batch start each binary classifier with different
    # initial hyperparameters for faster training
    if batchstart:
        if unique_labels_sample.shape[0] == 2:
            initial_hyperparams = [10, 0.1, 0.1]
        elif multimethod == 'OVA':
            initial_hyperparams = [      [6.8492834685818949, 0.14477735405874059, 0.11505973940953167, 0.16784579685852091, 0.14579818470438116, 0.2126379583720453] , \
                                         [1.3181854532161061, 0.09258211993254517, 0.11450045228885905, 0.11790388819451199, 0.1086982479299507, 0.10245724216413707] , \
                                         [9.7791686090410455, 0.22374784076872187, 0.12267756127614503, 0.15689385039161796, 0.15053194893192773, 0.16407979658658001] , \
                                         [2.2582603040726204, 0.094335396548258302, 0.10715020966770787, 0.12159803938262484, 0.10602973570769632, 0.11049244354804851] , \
                                         [7.2938984399015983, 0.17596808094257477, 0.11491604478166939, 0.15119995173504364, 0.13958638043376786, 0.22326110562350768] , \
                                         [1.4077139446315616, 0.089607506238812126, 0.10453780077599058, 0.11166052180703372, 0.1150483416526733, 0.11857137482213397] , \
                                         [2.2060009938736869, 0.10241659394335932, 0.11303756237146793, 0.10768410957954234, 0.11785833812467743, 0.10747534953984815] , \
                                         [7.6554851440019531, 0.22382188680747869, 0.11516948355600251, 0.15274002671000764, 0.13768689976809698, 0.17652786423194708] , \
                                         [1.0623261589420974, 0.097050383585362138, 0.11199055715072116, 0.10109216470144278, 0.11692650708295543, 0.10738201009336759] , \
                                         [2.1007082619712083, 0.10229248868420268, 0.1006613738506623, 0.12024688172009595, 0.1017077065314376, 0.11665487382492339] , \
                                         [6.6478790434690058, 0.16402450205336372, 0.10951458991331431, 0.15348740119481158, 0.13623702470667703, 0.21737397244777479] , \
                                         [6.8025281954962846, 0.21672889992125294, 0.11590052893923805, 0.17043604268458704, 0.13685361646970076, 0.14956547321693664] , \
                                         [7.2021158009345179, 0.21826504322487816, 0.11777393340596663, 0.15106113346516861, 0.14067226173489075, 0.17441444342162529] , \
                                         [6.215228961686841, 0.14699405518731287, 0.11933884582679528, 0.16736943342782895, 0.1421759558956755, 0.20772026330645485] , \
                                         [6.8656058871459358, 0.14586155295444572, 0.11541498593225026, 0.21253314132809673, 0.14500100712799271, 0.1679293236112373] , \
                                         [9.8517922483773628, 0.22298789683018275, 0.12227721088963638, 0.15789826477250088, 0.15116214340027495, 0.16302850785498746] , \
                                         [5.2272797159926947, 0.56354443728562975, 0.43930686698721211, 0.82201168824644344, 0.99138717882224914, 0.42977019748043727] ]

            # initial_hyperparams = [      [399.05308562273723, 1.1626890983021034, 11.057620223047245, 6.008405994132195, 9.0817575776979513, 13.899793403684312] , \
            #                              [3.9049433452381388, 3.4477798764588794, 3.6834907323124946, 3.1638814294727382, 1.2003489191663976, 3.8490589684483938] , \
            #                              [6.1983205523058711, 12.001229589617758, 6.6855766813102884, 4.7755879239116519, 9.9829246983761788, 15.764859618598203] , \
            #                              [3.7846762120107869, 0.8496977875404238, 3.5856900258094919, 3.3308564119107738, 2.8367347070492799, 2.7776117008801573] , \
            #                              [9.8224224556525783, 0.68771324750828899, 8.9490588226383956, 8.5707607552869298, 5.6550141640404279, 10.612547411758507] , \
            #                              [2.7558303832776696, 1.7954878854085916, 2.0617502662287603, 3.5930008475256563, 2.1311090224899791, 2.6274675430890198] , \
            #                              [2.3336699484643351, 0.90196078153363857, 1.4733898894093542, 1.582099060319307, 1.9247573440337757, 1.3181219261275474] , \
            #                              [2.3234805518013553, 2.0885474170008584, 1.7271669642137126, 1.4953900709784282, 1.4045693375833486, 2.0671591753518079] , \
            #                              [1.4724505050987888, 0.78616109378468646, 0.9821885399700605, 0.97611050616558137, 1.3818733133023293, 1.5773337390165068] , \
            #                              [7.2711722857067231, 1.2215472047715199, 5.8881954816154956, 7.104914279782002, 4.0605049662663202, 7.5552984410164017] , \
            #                              [2.5674440835588626, 2.8935381712981636, 2.0672422461452697, 3.1473143622661088, 1.747201113677721, 1.5801685454009098] , \
            #                              [3.0726351612382494, 1.5328236766496599, 4.1007581182696802, 3.8791592119088554, 3.0007265426296139, 3.2688133710706206] , \
            #                              [2.4606819987888953, 2.8583175442800179, 1.9883189513127952, 2.4110344154759629, 1.5622317191012578, 2.0264390690899292] , \
            #                              [5.7265833778953006, 2.0836582747942467, 4.5502979081177424, 5.8121346633504194, 6.4797823047740817, 6.3236927747995972] , \
            #                              [718.11185381141695, 1.2626806249002931, 5.2874209278331117, 16.827815365376928, 11.179633295870836, 12.97469466684799] , \
            #                              [3.7450324555704495, 2.3761629129163531, 2.9022000532835888, 2.1708805052764899, 2.0547427642740828, 2.353080594993604] ]
        elif multimethod == 'AVA':
            initial_hyperparams = [ [14.967, 0.547, 0.402],  \
                                    [251.979, 1.583, 1.318], \
                                    [420.376, 1.452, 0.750], \
                                    [780.641, 1.397, 1.682], \
                                    [490.353, 2.299, 1.526], \
                                    [73.999, 1.584, 0.954]]
        else:
            raise ValueError
        batch_config = gp.batch_start(optimiser_config, initial_hyperparams)
        logging.info('Using Batch Start Configuration')
        
    else:
        batch_config = optimiser_config
    logging.info('There are %d unique labels' % unique_labels_sample.shape[0])
    
    # Obtain the response function
    responsefunction = gp.classifier.responses.get(responsename)

    # Train the classifier!
    logging.info('Learning...')
    learned_classifier = gp.classifier.learn(
        training_features_sample_whiten, training_labels_sample, 
        kerneldef, responsefunction, batch_config, 
        multimethod = multimethod, approxmethod = approxmethod, 
        train = train, ftol = 1e-10)

    # Print the learnt kernel with its hyperparameters
    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(print_function, 
        learned_classifier, unique_labels_sample)

    # Print the matrix of learned classifier hyperparameters
    logging.info('Matrix of learned hyperparameters')
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)

    """Classifier Prediction"""

    if generate_draw:
        predictors = gp.classifier.query(learned_classifier, 
            query_features_sample_whiten)
        logging.info('Cached Predictor')
        query_latent_exp = gp.classifier.expectance(learned_classifier, 
            predictors)
        logging.info('Computed Expectance')
        query_latent_cov = gp.classifier.covariance(learned_classifier, 
            predictors)
        logging.info('Computed Covariance')
        query_draws = gp.classifier.draws(n_draws, 
            query_latent_exp, query_latent_cov, learned_classifier)
        logging.info('Sampled Draws')
        query_labels_prob = gp.classifier.predict_from_latent(query_latent_exp, 
            query_latent_cov, learned_classifier, fusemethod = fusemethod)
    else:
        query_labels_prob = gp.classifier.predict(
            query_features_sample_whiten, learned_classifier, 
            fusemethod = fusemethod)
    logging.info('Computed Prediction Probabilities')
    query_labels_pred = gp.classifier.classify(query_labels_prob, 
                                                unique_labels_sample)
    logging.info('Computed Prediction')
    query_labels_entropy = gp.classifier.entropy(query_labels_prob)    
    logging.info('Computed Prediction Entropy')

    """Visualise Query Prediction and Entropy"""
    fig = plt.figure(figsize = (19.2, 10.8))
    plt.scatter(
        query_locations_sample[:, 0], query_locations_sample[:, 1], 
        marker = 'x', c = query_labels_pred, vmin = unique_labels_sample[0], 
        vmax = unique_labels_sample[-1], cmap = mycmap)
    plt.title('Query Predictions')
    plt.xlabel('x [Eastings (m)]')
    plt.ylabel('y [Northings (m)]')
    cbar = plt.colorbar()
    cbar.set_label('Habitat Labels')
    cbar.set_ticks(unique_labels_sample)
    cbar.set_ticklabels(unique_labels_sample)
    if vis_fix_range:
        plt.xlim((vis_x_min, vis_x_max))
        plt.ylim((vis_y_min, vis_y_max))
    plt.gca().set_aspect('equal', adjustable = 'box')

    fig = plt.figure(figsize = (19.2, 10.8))
    plt.scatter(
        query_locations_sample[:, 0], query_locations_sample[:, 1], 
        marker = 'x', c = query_labels_entropy, cmap = mycmap)
    plt.title('Query Entropy')
    plt.xlabel('x [Eastings (m)]')
    plt.ylabel('y [Northings (m)]')
    cbar = plt.colorbar()
    cbar.set_label('Prediction Entropy')
    if vis_fix_range:
        plt.xlim((vis_x_min, vis_x_max))
        plt.ylim((vis_y_min, vis_y_max))
    plt.gca().set_aspect('equal', adjustable = 'box')

    """Visualise Query Draws"""

    if generate_draw:
        fig = plt.figure(figsize = (19.2, 10.8))
        for i in range(n_draws):
            plt.subplot(rows_subplot, cols_subplot, i + 1)
            plt.scatter(
                query_locations_sample[:, 0], query_locations_sample[:, 1], 
                marker = 'x', c = query_draws, cmap = mycmap)
            plt.title('Query Label Draws')
            plt.xlabel('x [Eastings (m)]')
            plt.ylabel('y [Northings (m)]')
            cbar = plt.colorbar()
            cbar.set_label('Habitat Labels')
            cbar.set_ticks(unique_labels_sample)
            cbar.set_ticklabels(unique_labels_sample)
            if vis_fix_range:
                plt.xlim((vis_x_min, vis_x_max))
                plt.ylim((vis_y_min, vis_y_max))
            plt.gca().set_aspect('equal', adjustable = 'box')
        logging.info('Plotted Sample Query Draws')

    if SAVE_RESULTS:
        gp.classifier.utils.save_all_figures(full_directory)

    plt.show()

















def remove_nan_queries(query_locations_old, query_features_old):

    kq = query_features_old.shape[1]

    query_locations_new = query_locations_old.copy()
    query_features_new = query_features_old.copy()

    valid_indices = ~np.isnan(query_features_new.mean(axis = 1))

    query_locations_new = query_locations_new[valid_indices]
    query_features_new = query_features_new[valid_indices]

    assert ~np.any(np.isnan(query_features_new))
    logging.debug('removed nan queries.')

    return query_locations_new, query_features_new

def load(directory_data, filename_training_data, filename_query_points):
    """Loads training and query data"""

    assert directory_data[-1] == '/'
    assert filename_training_data[-4:] == '.npz'
    assert filename_query_points[-4:] == '.npz'

    directory_training_data = directory_data + filename_training_data
    directory_query_points = directory_data + filename_query_points
    directory_query_points_clean = directory_data + \
                            filename_query_points.split('.')[0] + '_clean.npz'

    training_data = np.load(directory_training_data)

    logging.info('loading training locations...')
    training_locations = training_data['locations']
    logging.info('loading training labels...')
    training_labels = training_data['labels']
    logging.info('loading training features...')
    training_features = training_data['features']

    if os.path.isfile(directory_query_points_clean):

        query_data = np.load(directory_query_points_clean)

        logging.info('loading query locations...')
        query_locations_raw = query_data['locations']
        logging.info('loading query features...')
        query_features_raw = query_data['features']

        query_locations = query_locations_raw
        query_features  = query_features_raw

    else:

        query_data = np.load(directory_query_points)

        logging.info('loading query locations...')
        query_locations_raw = query_data['locations']
        logging.info('loading query features...')
        query_features_raw = query_data['features']

        logging.info('removing nan queries...')
        (query_locations, query_features) = \
            remove_nan_queries(query_locations_raw, query_features_raw)

        logging.info('saving cleaned data to "%s"' % 
            directory_query_points_clean)
        np.savez(directory_query_points_clean, 
            locations = query_locations, features = query_features)

    logging.info('Data Loading Done.')

    return training_locations, training_features, training_labels, \
            query_locations, query_features

def sample(training_locations, training_features, training_labels, 
    query_locations, query_features, 
    n_train_sample = 1000, n_query_sample = 10000):

    """Sample Training Data"""
    n_train = training_locations.shape[0]
    indices_train_sample = np.random.choice(np.arange(n_train), 
                            size = n_train_sample, replace = False)

    training_locations_sample = training_locations[indices_train_sample]
    training_features_sample = training_features[indices_train_sample]
    training_labels_sample = training_labels[indices_train_sample]

    logging.info('Sampled Number of Training Points: %d' % n_train_sample)

    """Sample Query Data"""
    n_query = query_locations.shape[0]
    indices_query_sample = np.random.choice(np.arange(n_query), 
                            size = n_query_sample, replace = False)

    query_locations_sample = query_locations[indices_query_sample]
    query_features_sample = query_features[indices_query_sample]

    logging.info('Sampled Number of Query Points: %d' % n_query_sample)

    """Loading and Sampling Assertions"""
    assert ~np.any(np.isnan(query_features))
    assert ~np.any(np.isnan(query_features_sample))   

    return  training_locations_sample, training_features_sample, \
            training_labels_sample, \
            query_locations_sample, query_features_sample

def cross_validation(training_locations, training_features, training_labels, 
    n_sample = 200):

    (training_locations_sample, training_features_sample, 
    training_labels_sample, 
    _, _) \
        = sample(training_locations, training_features, training_labels, 
                [0], [0], 
                n_train_sample = n_sample, n_query_sample = 0)

    k_location = 2
    k_features = 5

    indices = np.arange(n_sample)

    for i_sample in indices:

        i_sample_not = indices[indices != i_sample]

        X = training_features_sample[i_sample_not]
        y = training_labels_sample[i_sample_not]
        Xq = training_features_sample[[i_sample]]
        yq = training_labels_sample[[i_sample]]

        y_unique = np.unique(y)

        logging.info('Sample %d: Applying whitening on training and query features...'\
                        % i_sample)
        (Xw, whiten_params) = pre.standardise(X)
        Xqw = pre.standardise(Xq, params = whiten_params)

        logging.info('\tWhitening Parameters:\n\t')
        logging.info(whiten_params)

        # Training
        logging.info('Sample %d: Begin training for cross validation' % i_sample)

        optimiser_config = gp.LearningParams()
        optimiser_config.sigma = gp.auto_range(kerneldef)
        optimiser_config.walltime = walltime
        start_time = time.clock()
        learned_classifier = gp.classifier.learn(Xw, y, 
            kerneldef, responsefunction, optimiser_config, 
            approxmethod = approxmethod, train = True, ftol = 1e-10, 
            multimethod = multimethod)
        end_time = time.clock()
        learning_time = end_time - start_time
        logging.info('Sample %d: Learning Time: %f' % (i_sample, learning_time))

        # Print the learnt kernel with its hyperparameters
        print_function = gp.describer(kerneldef)
        print_learned_kernels(print_function, learned_classifier, y_unique)

        # Prediction
        yq_prob = gp.classifier.predict(Xqw, learned_classifier,
                                fusemethod = fusemethod)
        yq_pred = gp.classifier.classify(yq_prob, y_unique)
        yq_entropy = gp.classifier.entropy(yq_prob)

if __name__ == "__main__":
    main()
