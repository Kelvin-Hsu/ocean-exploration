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