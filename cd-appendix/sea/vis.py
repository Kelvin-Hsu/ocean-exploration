"""
Informative Seafloor Exploration
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def color_center_min_max(y):

    y_min = y.min()
    y_max = y.max()
    y_mean = y.mean()

    y_half = np.min([y_max - y_mean, y_mean - y_min])

    return y_mean - y_half, y_mean + y_half

def zero_center_min_max(y):

    y_min = y.min()
    y_max = y.max()

    if (y_max <= 0) or (y_min >= 0):
        return color_center_min_max(y)

    y_half = np.min([y_max, -y_min])
    return -y_half, y_half

def scatter(*args, colorcenter = 'none', **kwargs):
    if colorcenter == 'mean':
        vmin, vmax = color_center_min_max(kwargs.get('c'))
        kwargs.update({'vmin': vmin, 'vmax': vmax})
    elif colorcenter == 'zero':
        vmin, vmax = zero_center_min_max(kwargs.get('c'))
        kwargs.update({'vmin': vmin, 'vmax': vmax})
    return plt.scatter(*args, **kwargs)

def plot(*args, colorcenter = 'none', **kwargs):
    if colorcenter == 'mean':
        vmin, vmax = color_center_min_max(kwargs.get('c'))
        kwargs.update({'vmin': vmin, 'vmax': vmax})
    elif colorcenter == 'zero':
        vmin, vmax = zero_center_min_max(kwargs.get('c'))
        kwargs.update({'vmin': vmin, 'vmax': vmax})
    return plt.plot(*args, **kwargs)

def describe_plot(title = '', xlabel = '', ylabel = '', clabel = '', 
    cticks = None, cticklabels = None, fontsize = 40, 
    fontname = 'Helvetica', ticksize = 14, vis_range = None, 
    aspect_equal = True, axis_scale = 1):

    plt.gca().set_title(title, fontsize = fontsize, fontname = fontname)
    plt.gca().set_xlabel(xlabel, fontsize = fontsize, fontname = fontname)
    plt.gca().set_ylabel(ylabel, fontsize = fontsize, fontname = fontname)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    if clabel:
        cbar = plt.colorbar()
        cbar.set_label(clabel, fontsize = fontsize, fontname = fontname)
    if cticks is not None:
        cbar.set_ticks(cticks)
    if cticklabels is not None:
        cbar.set_ticklabels(cticklabels)
    if vis_range is not None:
        plt.gca().set_xlim(vis_range[:2])
        plt.gca().set_ylim(vis_range[2:])

    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/axis_scale))
    plt.gca().xaxis.set_major_formatter(ticks)
    plt.gca().yaxis.set_major_formatter(ticks)

    if aspect_equal:
        plt.gca().set_aspect('equal', adjustable = 'box')

def fig_size(fig_width_pt):
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt# width in inches
    fig_height = fig_width * golden_mean    # height in inches
    return fig_width, fig_height

def savefig(fig, filename):

    directory = '/'.join(filename.split('/')[0:-1]) + '/'

    if not os.path.isdir(directory):
        os.mkdir(directory)

    fig.savefig(filename)

def split_array(arr, num):

    n = arr.shape[0]
    extra = n % num
    arr_new = np.nan * np.ones(n - extra + num)
    arr_new[:n] = arr
    return np.reshape(arr_new, (arr_new.shape[0]/num, num))