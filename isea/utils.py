"""
Receding Horizon Informative Exploration

Utilities for receding horizon informative exploration
"""
import numpy as np
import logging
import os
def generate_line_path(x_s, x_f, n_points = 10):
    """
    Generates 'n_points' 2D coordinates from 'x_s' to 'x_f'
    """
    p = x_f - x_s
    r = np.linspace(0, 1, num = n_points)
    return np.outer(r, p) + x_s

def generate_line_paths(X_s, X_f, n_points = 10):
    """
    Generates 'n_points' 2D coordinates from multiple points 'X_s' to 'X_f'
    """

    assert X_s.shape == X_f.shape

    if hasattr(n_points, '__iter__'):

        assert n_points.shape[0] == X_s.shape[0]
        X = np.array([generate_line_path(X_s[i], X_f[i], n_points[i]) for i in range(X_s.shape[0])])
        return X.reshape(X.shape[0] * X.shape[1], X.shape[2])

    else:

        X = np.array([generate_line_path(X_s[i], X_f[i], n_points) for i in range(X_s.shape[0])])
        return X.reshape(X.shape[0] * X.shape[1], X.shape[2])

def generate_tracks(r_start, r_track, n_tracks, n_points, perturb_deg_scale = 5.0):

    thetas = np.random.uniform(0, 2*np.pi, size = (n_tracks,))

    thetas_perturb = thetas + np.random.normal(
        loc = 0.0, scale = np.deg2rad(perturb_deg_scale), size = (n_tracks,))

    r_starts = np.random.uniform(0, r_start, size = (n_tracks,))

    x1_starts = r_starts * np.cos(thetas_perturb)
    x2_starts = r_starts * np.sin(thetas_perturb)

    x_starts = np.array([x1_starts, x2_starts]).T

    r_tracks = np.random.uniform(0, r_track, size = (n_points, n_tracks))

    x1 = (x1_starts + r_tracks * np.cos(thetas)).flatten()
    x2 = (x2_starts + r_tracks * np.sin(thetas)).flatten()

    return np.array([x1, x2]).T

import matplotlib.pyplot as plt

def colorcenterminmax(y):

    y_min = y.min()
    y_max = y.max()
    y_mean = y.mean()

    y_half = np.min([y_max - y_mean, y_mean - y_min])

    return y_mean - y_half, y_mean + y_half

def scatter(*args, colorcenter = False, **kwargs):
    if colorcenter:
        vmin, vmax = colorcenterminmax(kwargs.get('c'))
        kwargs.update({'vmin': vmin, 'vmax': vmax})
    return plt.scatter(*args, **kwargs)

def plot(*args, colorcenter = False, **kwargs):
    if colorcenter:
        vmin, vmax = colorcenterminmax(kwargs.get('c'))
        kwargs.update({'vmin': vmin, 'vmax': vmax})
    return plt.plot(*args, **kwargs)

def describe_plot(title = '', xlabel = '', ylabel = '', clabel = '', 
    cticks = None, fontsize = 24, ticksize = 14, 
    vis_range = None, aspect_equal = True):

    plt.title(title, fontsize = fontsize)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize) 
    if clabel:
        cbar = plt.colorbar()
        cbar.set_label(clabel, fontsize = fontsize)
    if cticks is not None:
        cbar.set_ticks(cticks)
        cbar.set_ticklabels(cticks)
    if vis_range is not None:
        plt.xlim(vis_range[:2])
        plt.ylim(vis_range[2:])
    if aspect_equal:
        plt.gca().set_aspect('equal', adjustable = 'box')



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

def sample(X, F, y, Xq, Fq, n_train_sample = 1000, n_query_sample = 10000):

    """Sample Training Data"""
    n_train = X.shape[0]
    i_train_sample = np.random.choice(np.arange(n_train), 
                            size = n_train_sample, replace = False)

    X_sample = X[i_train_sample]
    F_sample = F[i_train_sample]
    y_sample = y[i_train_sample]

    logging.info('Sampled Number of Training Points: %d' % n_train_sample)

    """Sample Query Data"""
    n_query = Xq.shape[0]
    i_query_sample = np.random.choice(np.arange(n_query), 
                            size = n_query_sample, replace = False)

    Xq_sample = Xq[i_query_sample]
    Fq_sample = Fq[i_query_sample]

    logging.info('Sampled Number of Query Points: %d' % n_query_sample)

    """Loading and Sampling Assertions"""
    assert ~np.any(np.isnan(Fq))
    assert ~np.any(np.isnan(Fq_sample))

    return  X_sample, F_sample, y_sample, Xq_sample, Fq_sample