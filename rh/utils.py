"""
Receding Horizon Informative Exploration

Utilities for receding horizon informative exploration
"""
import numpy as np

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