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
import time

import logging
import sys
import os

logging.basicConfig(stream = sys.stdout, level = logging.DEBUG)

def remove_nan_queries(query_locations_old, query_features_old):

    kq = query_features_old.shape[1]

    # Initialise copies of 
    query_locations_new = query_locations_old.copy()
    query_features_new = query_features_old.copy()

    for ikq in range(kq):

        q_feature = query_features_new[:, ikq]
        query_locations_new = query_locations_new[~np.isnan(q_feature)]
        query_features_new = query_features_new[~np.isnan(q_feature)]

    logging.debug('removed nan queries.')

    return query_locations_new, query_features_new

def kerneldef(h, k):

    # Define the kernel used in the classifier
    return 	h(1e-3, 1e3, 1)*k('gaussian', 
            [h(1e-3, 1e3, 1), h(1e-3, 1e3, 1), h(1e-3, 1e3, 1), 
            h(1e-3, 1e3, 1), h(1e-3, 1e3, 1)])

def main():

    """
    Model Options
    """

    method = 'AVA' # 'AVA' or 'OVA', ignored for binary problem
    fusemethod = 'EXCLUSION' # 'MODE' or 'EXCLUSION', ignored for binary 

    n_train_sample = 500
    n_query_sample = 5000

    mycmap = cm.jet

    vis_fix_range = True
    vis_x_min = 365000
    vis_x_max = 390000
    vis_y_min = 8430000
    vis_y_max = 8448000

    """
    Load Data
    """

    # directory_data = 'Y:/BigDataKnowledgeDiscovery/AUV_data/'
    directory_data = 'C:/Users/kkeke_000/Dropbox/Thesis/Data/'
    directory_bathymetry_raw_data = directory_data + \
    	'scott_reef_wrangled_bathymetry3.pkl'
    directory_training_data = directory_data + 'bathAndLabels_10.npz'
    directory_query_points = directory_data + 'queryPoints.npz'
    directory_query_points_clean = directory_data + 'queryPointsClean.npz'

    training_data = np.load(directory_training_data)

    print('loading training locations...')
    training_locations = training_data['locations']
    print('loading training labels...')
    training_labels = training_data['labels']
    print('loading training features...')
    training_features = training_data['features']

    if os.path.isfile(directory_query_points_clean):

        query_data = np.load(directory_query_points)

        print('loading query locations...')
        query_locations_raw = query_data['locations']
        print('loading query features...')
        query_features_raw = query_data['query']

        query_locations = query_locations_raw
        query_features  = query_features_raw

    else:

        query_data = np.load(directory_query_points)

        print('loading query locations...')
        query_locations_raw = query_data['locations']
        print('loading query features...')
        query_features_raw = query_data['query']

        print('removing nan queries...')
        (query_locations, query_features) = \
            remove_nan_queries(query_locations_raw, query_features_raw)

        np.savez(directory_query_points_clean, 
            locations = query_locations, query = query_features)

    print('Data Loading Done.')

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
    Sample Training Data
    """
    indices_train_sample = np.random.choice(np.arange(n_train), 
                            size = n_train_sample, replace = False)

    training_locations_sample = training_locations[indices_train_sample]
    training_features_sample = training_features[indices_train_sample]
    training_labels_sample = training_labels[indices_train_sample]

    unique_labels_sample = np.unique(training_labels_sample)

    print('Sampled Number of Training Points: %d' % n_train_sample)

    """
    Sample Query Data
    """
    indices_query_sample = np.random.choice(np.arange(n_query), 
                            size = n_query_sample, replace = False)

    query_locations_sample = query_locations[indices_query_sample]
    query_features_sample = query_features[indices_query_sample]

    print('Sampled Number of Query Points: %d' % n_query_sample)

    """
    Visualise Sampled Training and Query Locations
    """

    fig = plt.figure()
    plt.scatter(
        training_locations_sample[:, 0], training_locations_sample[:, 1], 
        marker = 'x', c = training_labels_sample, cmap = mycmap)
    plt.title('Training Labels')
    plt.xlabel('x [Eastings (m)]')
    plt.ylabel('y [Northings (m)]')
    cbar = plt.colorbar()
    cbar.set_ticks(unique_labels_sample)
    cbar.set_ticklabels(unique_labels_sample)
    if vis_fix_range:
        plt.xlim((vis_x_min, vis_x_max))
        plt.ylim((vis_y_min, vis_y_max))

    fig = plt.figure()
    plt.scatter(
        query_locations_sample[:, 0], query_locations_sample[:, 1], 
        marker = 'x', c = [0.5, 0, 1], cmap = mycmap)
    plt.title('Query Locations')
    plt.xlabel('x [Eastings (m)]')
    plt.ylabel('y [Northings (m)]')
    if vis_fix_range:
        plt.xlim((vis_x_min, vis_x_max))
        plt.ylim((vis_y_min, vis_y_max))

    plt.show(block = False)

    """
    Visualise Sampled Features at Query Locations
    """

    for k in range(k_features):
        fig = plt.figure()
        plt.scatter(
            query_locations_sample[:, 0], query_locations_sample[:, 1], 
            marker = 'x', c = query_features_sample[:, k], cmap = mycmap)
        plt.title('Feature: %s at Query Points' % feature_names[k])
        plt.xlabel('x [Eastings (m)]')
        plt.ylabel('y [Northings (m)]')
        cbar = plt.colorbar()
        if vis_fix_range:
            plt.xlim((vis_x_min, vis_x_max))
            plt.ylim((vis_y_min, vis_y_max))

    plt.show(block = False)    

    """
    Classifier Training
    """

    # Compose the kernel defn into a callable function
    my_kernel_function = gp.compose(kerneldef)
    my_print_function = gp.describer(kerneldef)

    # Training
    print('===Begin Classifier Training===')
    print('Number of training points: %d' % n_train_sample)

    learning_hyperparams = gp.LearningParams()
    learning_hyperparams.sigma = gp.auto_range(kerneldef)
    learning_hyperparams.walltime = 100.0
    start_time = time.clock()
    learned_classifier = gp.classifier.learn(
        training_features_sample, training_labels_sample, 
        kerneldef, gp.classifier.responses.logistic, learning_hyperparams, 
        train = True, ftol = 1e-10, method = method)
    logging.info('Learning Time: %f' % (time.clock() - start_time))

    # Print the learnt kernel with its hyperparameters
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
            kernel_descriptions += 'Final Kernel %s: %s\n' \
                % (descript, my_print_function(
                    [learned_classifier[i_results].hyperparams]))
    else:
        kernel_descriptions = 'Final Kernel: %s\n' \
            % (my_print_function([learned_classifier.hyperparams]))
    print(kernel_descriptions)

    """
    Classifier Prediction
    """

    query_labels_prob = gp.classifier.predict(
                            query_features_sample, learned_classifier, 
                            fusemethod = fusemethod)

    query_labels_pred = gp.classifier.classify(query_labels_prob, 
                                                unique_labels_sample)

    query_labels_entropy = gp.classifier.entropy(query_labels_prob)    

    """
    Visualise Query Prediction and Entropy
    """

    fig = plt.figure()
    plt.scatter(
        query_locations_sample[:, 0], query_locations_sample[:, 1], 
        marker = 'x', c = query_labels_pred, vmin = unique_labels_sample[0], 
        vmax = unique_labels_sample[-1], cmap = mycmap)
    plt.title('Query Predictions')
    plt.xlabel('x [Eastings (m)]')
    plt.ylabel('y [Northings (m)]')
    cbar = plt.colorbar()
    cbar.set_ticks(unique_labels_sample)
    cbar.set_ticklabels(unique_labels_sample)
    if vis_fix_range:
        plt.xlim((vis_x_min, vis_x_max))
        plt.ylim((vis_y_min, vis_y_max))

    fig = plt.figure()
    plt.scatter(
        query_locations_sample[:, 0], query_locations_sample[:, 1], 
        marker = 'x', c = query_labels_entropy, cmap = mycmap)
    plt.title('Query Entropy')
    plt.xlabel('x [Eastings (m)]')
    plt.ylabel('y [Northings (m)]')
    cbar = plt.colorbar()
    if vis_fix_range:
        plt.xlim((vis_x_min, vis_x_max))
        plt.ylim((vis_y_min, vis_y_max))

    plt.show()

if __name__ == "__main__":
    main()
