"""
Informative Seafloor Exploration
"""
import matplotlib.pyplot as plt
import numpy as np

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