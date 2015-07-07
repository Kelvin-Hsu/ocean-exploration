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

logging.basicConfig(stream = sys.stdout, level = logging.DEBUG)

DEBUG = True

"""
Model Options
"""

SAVE_FIGURES = True

method = 'OVA' # 'AVA' or 'OVA', ignored for binary problem
fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary
responsename = 'probit'
approxmethod = 'pls'

n_train_sample = 200
n_query_sample = 10000
walltime = 900

mycmap = cm.jet

vis_fix_range = True
vis_x_min = 365000
vis_x_max = 390000
vis_y_min = 8430000
vis_y_max = 8448000

if responsename == 'probit':
    responsefunction = gp.classifier.responses.probit
elif responsename == 'logistic':
    responsefunction = gp.classifier.responses.logistic
else:
    raise ValueError

def kerneldef(h, k):

    # Define the kernel used in the classifier
    return  h(1e-3, 1e3, 10)*k('gaussian', 
            [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), 
            h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])

def main():

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
            print('\t\tTraining:', training_features_sample[:, k].min(), training_features_sample[:, k].max())
            print('\t\tQuery:', query_features_sample[:, k].min(), query_features_sample[:, k].max())

            print('\tWhiten')
            print('\t\tTraining:', training_features_sample_whiten[:, k].min(), training_features_sample_whiten[:, k].max())
            print('\t\tQuery:', query_features_sample_whiten[:, k].min(), query_features_sample_whiten[:, k].max())

    # plt.show(block = False)

    """
    Classifier Training
    """

    # Training
    print('===Begin Classifier Training===')
    print('Number of training points: %d' % n_train_sample)

    learning_hyperparams = gp.LearningParams()
    learning_hyperparams.sigma = gp.auto_range(kerneldef)
    learning_hyperparams.walltime = walltime
    start_time = time.clock()
    learned_classifier = gp.classifier.learn(
        training_features_sample_whiten, training_labels_sample, 
        kerneldef, responsefunction, learning_hyperparams, 
        approxmethod = approxmethod, train = True, ftol = 1e-10, 
        method = method)
    end_time = time.clock()
    learning_time = end_time - start_time
    print('Learning Time: %f' % learning_time)

    # Print the learnt kernel with its hyperparameters
    my_print_function = gp.describer(kerneldef)
    print_learned_kernels(my_print_function, learned_classifier, 
        unique_labels_sample)

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
        figure_directory_name = "C:/Users/kkeke_000/" \
        "Dropbox/Thesis/Results/ocean-exploration/"
        figure_sub_directory_name = "scott_reef__response_%s_approxmethod_%s" \
        "_training_%d_query_%d_walltime_%d_datetime_%s_method_%s_fusemethod_%s/" \
            % ( responsename, approxmethod, 
                n_train_sample, n_query_sample, walltime, 
                time.strftime("%Y%m%d_%H%M%S", time.gmtime()), 
                method, fusemethod)
        figure_full_directory_name = '%s%s' \
            % (figure_directory_name, figure_sub_directory_name)

        # Create directories
        if not os.path.isdir(figure_directory_name):
            os.mkdir(figure_directory_name)
        if not os.path.isdir(figure_full_directory_name):
            os.mkdir(figure_full_directory_name)

        # Go through each figure and save them
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig('%sFigure%d.png' % (figure_full_directory_name, i))

        print('Figures Saved')

        textfilename = '%slog.txt' % (figure_full_directory_name)
        textfile = open(textfilename, 'w')
        sys.stdout = textfile

        print('Learning Time: %f' % learning_time)
        print_learned_kernels(my_print_function, learned_classifier, 
            unique_labels_sample)

        textfilename.close()

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

def print_learned_kernels(my_print_function, learned_classifier, 
    unique_labels_sample):

    kernel_descriptions = ''
    if isinstance(learned_classifier, list):
        n_results = len(learned_classifier)
        for i_results in range(n_results):
            i_class = learned_classifier[i_results].cache.get('i_class')
            j_class = learned_classifier[i_results].cache.get('j_class')
            class1 = unique_labels_sample[i_class]
            if j_class == -1:
                class2 = 'all'
                descript = '(Labels %d v.s. %s)' % (class1, class2)
            else:
                class2 = unique_labels_sample[j_class]
                descript = '(Labels %d v.s. %d)' % (class1, class2)
            kernel_descriptions += 'Final Kernel %s: %s \t | \t '\
                'Log Marginal Likelihood: %.8f \n' \
                % (descript, my_print_function(
                    [learned_classifier[i_results].hyperparams]), 
                     learned_classifier[i_results].log_marginal_likelihood)
    else:
        kernel_descriptions = 'Final Kernel: %s\n' \
            % (my_print_function([learned_classifier.hyperparams]))
    print(kernel_descriptions)

def cross_validation(training_locations, training_features, training_labels, 
    n_sample):

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

        print('Set %d: Applying whitening on training and query features...' % i_sample)
        (Xw, whiten_params) = pre.standardise(X)
        Xqw = pre.standardise(Xq, params = whiten_params)

        print('\tWhitening Parameters:\n\t')
        print(whiten_params)

        # Training
        print('Set %d: Begin training for cross validation' % i_sample)

        learning_hyperparams = gp.LearningParams()
        learning_hyperparams.sigma = gp.auto_range(kerneldef)
        learning_hyperparams.walltime = walltime
        start_time = time.clock()
        learned_classifier = gp.classifier.learn(Xw, y, 
            kerneldef, responsefunction, learning_hyperparams, 
            approxmethod = approxmethod, train = True, ftol = 1e-10, 
            method = method)
        end_time = time.clock()
        learning_time = end_time - start_time
        print('Set %d: Learning Time: %f' % (i_sample, learning_time))

        # Print the learnt kernel with its hyperparameters
        my_print_function = gp.describer(kerneldef)
        print_learned_kernels(my_print_function, learned_classifier, y_unique)

        # Prediction
        yq_prob = gp.classifier.predict(
                                Xqw, learned_classifier,
                                fusemethod = fusemethod)

        yq_pred = gp.classifier.classify(yq_prob, y_unique)

        yq_entropy = gp.classifier.entropy(yq_prob)

# def cross_validation(training_locations, training_features, training_labels, 
#     n_size = 200, n_set = 5):

#     n_total = n_size * n_set

#     (training_locations_sample, training_features_sample, 
#     training_labels_sample, 
#     _, _) \
#         = sample(training_locations, training_features, training_labels, 
#                 [0], [0], 
#                 n_train_sample = n_total, n_query_sample = 0)

#     k_location = 2
#     k_features = 5
#     training_locations_sample.reshape(n_set, n_size, k_location)
#     training_features_sample.reshape(n_set, n_size, k_features)
#     training_labels_sample.reshape(n_set, n_size)

#     indices = np.arange(n_set)

#     for i_sample in indices:

#         i_sample_not = indices[indices != i_sample]

#         X = training_features_sample[i_sample_not].reshape((n_set - 1) * n_size, 
#                                                          k_features)
#         y = training_labels_sample[i_sample_not].reshape((n_set - 1) * n_size)

#         Xq = training_features_sample[i_sample]

#         yq = training_labels_sample[i_sample]

#         y_unique = np.unique(y)

#         print('Set %d: Applying whitening on training and query features...' % i_sample)
#         (Xw, whiten_params) = pre.standardise(X)
#         Xqw = pre.standardise(Xq, params = whiten_params)

#         print('\tWhitening Parameters:\n\t')
#         print(whiten_params)

#         # Training
#         print('Set %d: Begin training for cross validation' % i_sample)

#         learning_hyperparams = gp.LearningParams()
#         learning_hyperparams.sigma = gp.auto_range(kerneldef)
#         learning_hyperparams.walltime = walltime
#         start_time = time.clock()
#         learned_classifier = gp.classifier.learn(Xw, y, 
#             kerneldef, responsefunction, learning_hyperparams, 
#             approxmethod = approxmethod, train = True, ftol = 1e-10, 
#             method = method)
#         end_time = time.clock()
#         learning_time = end_time - start_time
#         print('Set %d: Learning Time: %f' % (i_sample, learning_time))

#         # Print the learnt kernel with its hyperparameters
#         my_print_function = gp.describer(kerneldef)
#         print_learned_kernels(my_print_function, learned_classifier, y_unique)

#         # Prediction
#         yq_prob = gp.classifier.predict(
#                                 Xqw, learned_classifier,
#                                 fusemethod = fusemethod)

#         yq_pred = gp.classifier.classify(yq_prob, y_unique)

#         yq_entropy = gp.classifier.entropy(yq_prob)


if __name__ == "__main__":
    main()
