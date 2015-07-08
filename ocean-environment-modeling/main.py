"""
Module  : Ocean Environment Modeling
File    : main.py

Author  : Kelvin Hsu
Date    : 1st July 2015

--Description--
	
    Task 1: Read data from pickle file
    Task 2: Apply GP Classification to investigate environment modeling
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

    # Define the kernel used in the classifier
    return  h(1e-3, 1e3, 10)*k('gaussian', 
            [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), 
            h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])

def main():

    """
    Model Options
    """
    DEBUG = True
    SAVE_FIGURES = True

    approxmethod = 'laplace' # 'laplace' or 'pls'
    method = 'OVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
    responsename = 'probit' # 'probit' or 'logistic'
    batchstart = True
    walltime = 300.0

    n_train_sample = 1500
    n_query_sample = 10000
    
    """
    Visualisation Options
    """
    mycmap = cm.jet

    vis_fix_range = True
    vis_x_min = 365000
    vis_x_max = 390000
    vis_y_min = 8430000
    vis_y_max = 8448000

    """
    Process Options
    """
    if DEBUG:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(stream = sys.stdout, level = logging_level)
    model_options = {   'approxmethod': approxmethod,
                        'method': method,
                        'fusemethod': fusemethod,
                        'responsename': responsename,
                        'batchstart': batchstart,
                        'walltime': walltime}
    """
    Load Data
    """
    (training_locations, training_features, training_labels, \
            query_locations, query_features) = load()

    n_train = training_features.shape[0]
    n_query = query_features.shape[0]
    k_features = training_features.shape[1]
    feature_names = [   'Bathymetry (Depth)', 
                        'Aspect (Short Scale)',
                        'Rugosity (Short Scale)',
                        'Aspect (Long Scale)',
                        'Rugosity (Long Scale)']

    print('Raw Number of Training Points: %d' % n_train)
    print('Raw Number of Query Points: %d' % n_query)

    """
    Sample Training Data and Query Points
    """

    (training_locations_sample, training_features_sample, 
    training_labels_sample, 
    query_locations_sample, query_features_sample) \
        = sample(training_locations, training_features, training_labels, 
        query_locations, query_features, 
        n_train_sample = n_train_sample, n_query_sample = n_query_sample)

    unique_labels_sample = np.unique(training_labels_sample)

    """
    Whiten the feature space
    """

    print('Applying whitening on training and query features...')
    (training_features_sample_whiten, whiten_params) = \
        pre.standardise(training_features_sample)

    query_features_sample_whiten = \
        pre.standardise(query_features_sample, params = whiten_params)

    print('Whitening Parameters:')
    print(whiten_params)

    """
    Visualise Sampled Training Locations
    """

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

    """
    Visualise Features at Sampled Query Locations
    """

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

        if DEBUG:
            print('Feature %d' % k)
            print('\tPre-Whiten')
            print('\t\tTraining:', training_features_sample[:, k].min(), 
                training_features_sample[:, k].max())
            print('\t\tQuery:', query_features_sample[:, k].min(), 
                query_features_sample[:, k].max())

            print('\tWhiten')
            print('\t\tTraining:', training_features_sample_whiten[:, k].min(), 
                training_features_sample_whiten[:, k].max())
            print('\t\tQuery:', query_features_sample_whiten[:, k].min(), 
                query_features_sample_whiten[:, k].max())

    # plt.show(block = False)

    """
    Classifier Training
    """

    # Training
    print('===Begin Classifier Training===')
    print('Number of training points: %d' % n_train_sample)

    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # User can choose to batch start each binary classifier with different
    # initial hyperparameters for faster training
    if batchstart:
        if unique_labels_sample.shape[0] == 2:
            initial_hyperparams = [10, 0.1, 0.1]
        elif method == 'OVA':
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
        elif method == 'AVA':
            initial_hyperparams = [ [14.967, 0.547, 0.402],  \
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

    print('Learning...')
    # Train the classifier!
    learned_classifier = gp.classifier.learn(
        training_features_sample_whiten, training_labels_sample, 
        kerneldef, responsefunction, optimiser_config, 
        approxmethod = approxmethod, train = True, ftol = 1e-10, 
        method = method)

    # Print the learnt kernel with its hyperparameters
    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(print_function, 
        learned_classifier, unique_labels_sample)

    # Print the matrix of learned classifier hyperparameters
    print('Matrix of learned hyperparameters')
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)

    """
    Classifier Prediction
    """

    query_labels_prob = gp.classifier.predict(
                            query_features_sample_whiten, learned_classifier, 
                            fusemethod = fusemethod)

    query_labels_pred = gp.classifier.classify(query_labels_prob, 
                                                unique_labels_sample)

    query_labels_entropy = gp.classifier.entropy(query_labels_prob)    

    """
    Visualise Query Prediction and Entropy
    """

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
    
    if SAVE_FIGURES:

        # Directory names
        home_directory = "C:/Users/kkeke_000/" \
        "Dropbox/Thesis/Results/ocean-exploration/"
        save_directory = "scott_reef__response_%s_approxmethod_%s" \
        "_training_%d_query_%d_walltime_%d" \
        "_method_%s_fusemethod_%s/" \
            % ( responsename, approxmethod, 
                n_train, n_query, walltime, 
                method, fusemethod)
        full_directory = gp.classifier.utils.save_all_figures(save_directory, 
            home_directory = home_directory, append_time = True)

        textfilename = '%slog.txt' % full_directory
        textfile = open(textfilename, 'w')
        sys.stdout = textfile

        gp.classifier.utils.print_learned_kernels(print_function, 
            learned_classifier, unique_labels_sample)

        textfile.close()

    plt.show()

















def remove_nan_queries(query_locations_old, query_features_old):

    kq = query_features_old.shape[1]

    # Initialise copies of 
    query_locations_new = query_locations_old.copy()
    query_features_new = query_features_old.copy()

    valid_indices = ~np.isnan(query_features_new.mean(axis = 1))

    query_locations_new = query_locations_new[valid_indices]
    query_features_new = query_features_new[valid_indices]

    assert ~np.any(np.isnan(query_features_new))
    logging.debug('removed nan queries.')

    return query_locations_new, query_features_new

def load():

    """
    Load Data
    """

    # directory_data = 'Y:/BigDataKnowledgeDiscovery/AUV_data/'
    directory_data = 'C:/Users/kkeke_000/Dropbox/Thesis/Data/'
    directory_bathymetry_raw_data = directory_data + \
        'scott_reef_wrangled_bathymetry3.pkl'
    directory_training_data = directory_data + 'training_data_unmerged.npz'
    directory_query_points = directory_data + 'query_points.npz'
    directory_query_points_clean = directory_data + 'query_points_clean.npz'

    training_data = np.load(directory_training_data)

    print('loading training locations...')
    training_locations = training_data['locations']
    print('loading training labels...')
    training_labels = training_data['labels']
    print('loading training features...')
    training_features = training_data['features']

    if os.path.isfile(directory_query_points_clean):

        query_data = np.load(directory_query_points_clean)

        print('loading query locations...')
        query_locations_raw = query_data['locations']
        print('loading query features...')
        query_features_raw = query_data['features']

        query_locations = query_locations_raw
        query_features  = query_features_raw

    else:

        query_data = np.load(directory_query_points)

        print('loading query locations...')
        query_locations_raw = query_data['locations']
        print('loading query features...')
        query_features_raw = query_data['features']

        print('removing nan queries...')
        (query_locations, query_features) = \
            remove_nan_queries(query_locations_raw, query_features_raw)

        print('saving cleaned data to "%s"' % directory_query_points_clean)
        np.savez(directory_query_points_clean, 
            locations = query_locations, features = query_features)

    print('Data Loading Done.')

    return training_locations, training_features, training_labels, \
            query_locations, query_features

def sample(training_locations, training_features, training_labels, 
    query_locations, query_features, 
    n_train_sample = 1000, n_query_sample = 10000):

    """
    Sample Training Data
    """
    n_train = training_locations.shape[0]
    indices_train_sample = np.random.choice(np.arange(n_train), 
                            size = n_train_sample, replace = False)

    training_locations_sample = training_locations[indices_train_sample]
    training_features_sample = training_features[indices_train_sample]
    training_labels_sample = training_labels[indices_train_sample]

    print('Sampled Number of Training Points: %d' % n_train_sample)

    """
    Sample Query Data
    """
    n_query = query_locations.shape[0]
    indices_query_sample = np.random.choice(np.arange(n_query), 
                            size = n_query_sample, replace = False)

    query_locations_sample = query_locations[indices_query_sample]
    query_features_sample = query_features[indices_query_sample]

    print('Sampled Number of Query Points: %d' % n_query_sample)

    """
    Loading and Sampling Assertions
    """

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

        print('Sample %d: Applying whitening on training and query features...'\
                        % i_sample)
        (Xw, whiten_params) = pre.standardise(X)
        Xqw = pre.standardise(Xq, params = whiten_params)

        print('\tWhitening Parameters:\n\t')
        print(whiten_params)

        # Training
        print('Sample %d: Begin training for cross validation' % i_sample)

        optimiser_config = gp.LearningParams()
        optimiser_config.sigma = gp.auto_range(kerneldef)
        optimiser_config.walltime = walltime
        start_time = time.clock()
        learned_classifier = gp.classifier.learn(Xw, y, 
            kerneldef, responsefunction, optimiser_config, 
            approxmethod = approxmethod, train = True, ftol = 1e-10, 
            method = method)
        end_time = time.clock()
        learning_time = end_time - start_time
        print('Sample %d: Learning Time: %f' % (i_sample, learning_time))

        # Print the learnt kernel with its hyperparameters
        print_function = gp.describer(kerneldef)
        print_learned_kernels(print_function, learned_classifier, y_unique)

        # Prediction
        yq_prob = gp.classifier.predict(
                                Xqw, learned_classifier,
                                fusemethod = fusemethod)

        yq_pred = gp.classifier.classify(yq_prob, y_unique)

        yq_entropy = gp.classifier.entropy(yq_prob)

if __name__ == "__main__":
    main()
