"""
Informative Seafloor Exploration
"""
import sys
import os
import numpy as np
import logging

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

def remove_nan_queries(Xq_old, Fq_old):

    kq = Fq_old.shape[1]

    Xq_new = Xq_old.copy()
    Fq_new = Fq_old.copy()

    valid_indices = ~np.isnan(Fq_new.mean(axis = 1))

    Xq_new = Xq_new[valid_indices]
    Fq_new = Fq_new[valid_indices]

    assert ~np.any(np.isnan(Fq_new))
    logging.debug('Removed all NaN queries.')

    return Xq_new, Fq_new

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
    X = training_data['locations']
    logging.info('loading training labels...')
    y = training_data['labels']
    logging.info('loading training features...')
    F = training_data['features']

    if os.path.isfile(directory_query_points_clean):

        query_data = np.load(directory_query_points_clean)

        logging.info('loading query locations...')
        Xq_raw = query_data['locations']
        logging.info('loading query features...')
        Fq_raw = query_data['features']

        Xq = Xq_raw
        Fq  = Fq_raw

    else:

        query_data = np.load(directory_query_points)

        logging.info('loading query locations...')
        Xq_raw = query_data['locations']
        logging.info('loading query features...')
        Fq_raw = query_data['features']

        logging.info('removing nan queries...')
        (Xq, Fq) = \
            remove_nan_queries(Xq_raw, Fq_raw)

        logging.info('saving cleaned data to "%s"' % 
            directory_query_points_clean)
        np.savez(directory_query_points_clean, 
            locations = Xq, features = Fq)

    logging.info('Data Loading Done.')

    return X, F, y, Xq, Fq

def sample(X, F, y, Xq, Fq, 
    n_train = 200, n_query = 10000, t_seed = None, q_seed = None, 
    features = None, unique_labels = False, unique_seed = None):
    """Sample Training Data"""
    assert n_train < 2000
    assert n_query < 250000

    if unique_labels:
        if unique_seed:
            np.random.seed(unique_seed)
            i_perm = np.random.permutation(np.arange(y.shape[0]))
            y_unique, i_train_sample_perm = np.unique(y[i_perm], 
                return_index = True)
            i_train_sample = i_perm[i_train_sample_perm]
        else:
            y_unique, i_train_sample = np.unique(y, return_index = True)
        n_unique = y_unique.shape[0]

        if n_train > n_unique:
            if t_seed:
                np.random.seed(t_seed)
            i_train_sample = np.append(i_train_sample, 
                np.random.choice(np.arange(X.shape[0]), 
                size = n_train - n_unique, replace = False))

    else:
        if t_seed:
            np.random.seed(t_seed)
        i_train_sample = np.random.choice(np.arange(X.shape[0]), 
                                size = n_train, replace = False)

    X_sample = X[i_train_sample]
    F_sample = F[i_train_sample]
    y_sample = y[i_train_sample]

    logging.info('Total Number of Trainint Points: %d' % X.shape[0])
    logging.info('Sampled Number of Training Points: %d' % X_sample.shape[0])

    """Sample Query Data"""
    if q_seed:
        np.random.seed(q_seed)
    i_query_sample = np.random.choice(np.arange(Xq.shape[0]), 
                            size = n_query, replace = False)

    Xq_sample = Xq[i_query_sample]
    Fq_sample = Fq[i_query_sample]

    logging.info('Total Number of Query Points: %d' % Xq.shape[0])
    logging.info('Sampled Number of Query Points: %d' % Xq_sample.shape[0])

    """Pick Features"""
    if features is not None:
        F_sample = F_sample[:, features]
        Fq_sample = Fq_sample[:, features]


    """Loading and Sampling Assertions"""
    assert ~np.any(np.isnan(Fq))
    assert ~np.any(np.isnan(Fq_sample))

    return  X_sample, F_sample, y_sample, Xq_sample, Fq_sample, \
            i_train_sample, i_query_sample

def load_ground_truth(filename, assert_query_seed = None):

    # filename = '%struthmodel.npz' % directory
    truthmodel = np.load(filename)

    if assert_query_seed is not None:
        logging.debug('Truth Model Query Seed: %d' % truthmodel['q_seed'])
        assert truthmodel['q_seed'] == assert_query_seed

    return truthmodel['yq_pred']