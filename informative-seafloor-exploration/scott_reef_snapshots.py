import numpy as np
import shelve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
import time
import matplotlib.ticker as ticker
import sea
from computers import gp
import os

def kerneldef5(h, k):
    """Define the kernel used in the classifier"""
    return  h(1e-4, 1e4, 10)*k('gaussian', 
            [h(1e-4, 1e4, 0.1), h(1e-4, 1e4, 0.1), h(1e-4, 1e4, 0.1), 
            h(1e-4, 1e4, 0.1), h(1e-4, 1e4, 0.1)])

def kerneldef3(h, k):
    """Define the kernel used in the classifier"""
    return  h(1e-4, 1e4, 10)*k('gaussian', 
            [h(1e-4, 1e4, 0.1), h(1e-4, 1e4, 0.1), h(1e-4, 1e4, 0.1)])

def main():

    logging.basicConfig(level = logging.DEBUG)

    main_directory = "../../../Results/scott-reef/"
    directory = main_directory + "loc1_new_20150827_072158__method_LDE_start_377500_8440000_hsteps30_horizon5000/"
    trials = np.arange(25, 201, 25)
    logging.info(trials)

    save_directory = directory + 'PostProcess/'
    os.mkdir(save_directory) if not os.path.exists(save_directory) else None

    data = load_data()
    for i in trials:
        logging.info('Saving for trial %d...' % i)
        filename = directory + "history%d.npz" % i
        npzfile = dict(np.load(filename))
        save_maps(*data, full_directory = save_directory, **npzfile)
        logging.info('Finished saving for trial %d' % i)

def load_data():

    """File Locations"""
    directory_data = '../../../Data/'
    filename_training_data = 'training_data_unmerged.npz'
    filename_query_points = 'query_points.npz'
    filename_truth = directory_data + 'truthmodel_t800_q100000_ts250_qs500.npz'
    filename_start = directory_data + 'finalmodel_t200_q100000_ts250_qs500'\
        '_method_LDE_start377500_8440000_hsteps30_horizon5000.npz'

    T_SEED = sea.io.parse('-tseed', 250)
    Q_SEED = sea.io.parse('-qseed', 500) 
    N_TRAIN = sea.io.parse('-ntrain', 200)
    N_QUERY = sea.io.parse('-nquery', 100000)
    i_features = [0, 1, 2, 3, 4]

    X, F, y, Xq, Fq, i_train, i_query = \
        sea.io.sample(*sea.io.load(directory_data, 
            filename_training_data, filename_query_points), 
            n_train = N_TRAIN, n_query = N_QUERY,
            t_seed = T_SEED, q_seed = Q_SEED, features = i_features)

    return X, F, y, Xq, Fq

def save_maps(  X, F, y, Xq, Fq,
                full_directory = '',
                yq_mie = None,
                yq_lde = None,
                yq_pred = None,
                miss_ratio_array = None,
                vis_range = None,
                xq1_nows = None,
                xq2_nows = None,
                yq_nows = None,
                xq1_path = None,
                xq2_path = None,
                yq_path = None,
                y_unique = None,
                mycmap = None,
                xq_now = None,
                yq_now = None,
                horizon = None,
                colorcenter_analysis = None,
                colorcenter_lde = None,
                r = None,
                i_trials = None,
                **kwargs):

    # Plot the current situation
    fig1 = plt.figure(1, figsize = (19.2, 10.8))
    fig2 = plt.figure(2, figsize = (19.2, 10.8))
    fig3 = plt.figure(3, figsize = (19.2, 10.8))
    fig4 = plt.figure(4, figsize = (19.2, 10.8))
    fig5 = plt.figure(5, figsize = (19.2, 10.8))

    mycmap = cm.get_cmap(name = 'jet', lut = None)
    horizon = 5000.0
    h_steps = 30
    r = horizon/h_steps
    colorcenter_lde = colorcenter_analysis
    FONTSIZE = 50
    FONTNAME = 'Sans Serif'
    TICKSIZE = 24
    SAVE_TRIALS = 25

    i_trials -= 1
    miss_ratio = miss_ratio_array[i_trials]

    y_names_all = [ 'None',
                    'Under-Exposed', 
                    'Under-Exposed',
                    'Barron Sand 1',
                    'Low Density Coral 1',
                    'Sand Biota 1',
                    'Low Density Coral 2',
                    'Dense Coral 1',
                    'Dense Coral 2',
                    'Dense Coral 3',
                    'Sand Biota 2',
                    'Low Density Coral 3',
                    'Low Density Coral 4',
                    'Patch 1',
                    'Patch 2',
                    'Patch 3',
                    'Barron Sand 2',
                    'Sand Biota 3',
                    'Over-Exposed',
                    'Barron Sand 3',
                    'Under-Exposed',
                    'Under-Exposed',
                    'Sand Biota 4',
                    'Misc',
                    'Under-Exposed']

    y_names = [y_names_all[i] for i in y_unique.astype(int)]

    """ Linearised Model Differential Entropy Map """
    logging.info('Saving Linearised Model Differential Entropy Map')

    # Prepare Figure 1
    plt.figure(fig1.number)
    plt.clf()
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_lde, s = 5, 
        cmap = cm.coolwarm, colorcenter = colorcenter_lde)
    sea.vis.describe_plot(title = 'Linearised Model Differential Entropy', 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Differential Entropy',
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)

    # Plot the path on top
    sea.vis.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
        facecolors = 'none', 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)
    sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
    sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)

    # Plot the horizon
    gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
        linewidth = 2, marker = '.')

    plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + r, 0, -r/4, 
        head_width = r/4, head_length = r/4, fc = 'k', ec = 'k')

    # Save the plot
    fig1.tight_layout()
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.savefig('%slde%d.png' 
        % (full_directory, i_trials + 1))
    if (i_trials == 0) or (((i_trials + 1) % SAVE_TRIALS) == 0):
        plt.savefig('%slde%d.eps' 
            % (full_directory, i_trials + 1))

    # Plot the proposed path
    sea.vis.scatter(xq1_path, xq2_path, c = yq_path, 
        s = 60, marker = 'D', 
        vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
    sea.vis.plot(xq1_path, xq2_path, c = 'k', linewidth = 2)

    # Save the plot
    fig1.tight_layout()
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.savefig('%slde_propose%d.png' 
        % (full_directory, i_trials + 1))
    if (i_trials == 0) or (((i_trials + 1) % SAVE_TRIALS) == 0):
        plt.savefig('%slde_propose%d.eps' 
            % (full_directory, i_trials + 1))

    """ True Entropy Map """
    logging.info('Saving Prediction Information Entropy Map')

    # Prepare Figure 3
    plt.figure(fig3.number)
    plt.clf()
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_mie, s = 5, 
        cmap = cm.coolwarm, colorcenter = colorcenter_analysis)
    sea.vis.describe_plot(title = 'Prediction Information Entropy', 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Information Entropy',
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)

    # Plot the path on top
    sea.vis.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
        facecolors = 'none', 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)
    sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
    sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)

    # Save the plot
    fig3.tight_layout()
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.savefig('%smie%d.png' 
        % (full_directory, i_trials + 1))
    if (i_trials == 0) or (((i_trials + 1) % SAVE_TRIALS) == 0):
        plt.savefig('%smie%d.eps' 
            % (full_directory, i_trials + 1))

    # Plot the horizon
    gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
        linewidth = 2, marker = '.')

    plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + r, 0, -r/4, 
        head_width = r/4, head_length = r/4, fc = 'k', ec = 'k')

    # Plot the proposed path
    sea.vis.scatter(xq1_path, xq2_path, c = yq_path, 
        s = 60, marker = 'D', 
        vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
    sea.vis.plot(xq1_path, xq2_path, c = 'k', linewidth = 2)

    # Save the plot
    fig3.tight_layout()
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.savefig('%smie_propose%d.png' 
        % (full_directory, i_trials + 1))
    if (i_trials == 0) or (((i_trials + 1) % SAVE_TRIALS) == 0):
        plt.savefig('%smie_propose%d.eps' 
            % (full_directory, i_trials + 1))

    """ Class Prediction Map """
    logging.info('Saving Class Prediction Map')

    # Prepare Figure 4
    plt.figure(fig4.number)
    plt.clf()
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        marker = 'x', c = yq_pred, s = 5, 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)
    sea.vis.describe_plot(
        title = 'Prediction Map [Miss Ratio: {0:.2f}{1}]'.format(
            100 * miss_ratio, '%'), 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)

    # Plot the path on top
    sea.vis.scatter(xq1_nows, xq2_nows, c = yq_nows, s = 60, 
        facecolors = 'none', 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)
    sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
    sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
        vmin = y_unique[0], vmax = y_unique[-1], 
        cmap = mycmap)

    # Plot the horizon
    gp.classifier.utils.plot_circle(xq_now[-1], horizon, c = 'k', 
        linewidth = 2, marker = '.')

    plt.gca().arrow(xq_now[-1][0], xq_now[-1][1] + r, 0, -r/4, 
        head_width = r/4, head_length = r/4, fc = 'k', ec = 'k')

    # Save the plot
    fig4.tight_layout()
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.savefig('%spred%d.png' 
        % (full_directory, i_trials + 1))
    if (i_trials == 0) or (((i_trials + 1) % SAVE_TRIALS) == 0):
        plt.savefig('%spred%d.eps' 
            % (full_directory, i_trials + 1))

    # Plot the proposed path
    sea.vis.scatter(xq1_path, xq2_path, c = yq_path, 
        s = 60, marker = 'D', 
        vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap)
    sea.vis.plot(xq1_path, xq2_path, c = 'k', linewidth = 2)

    # Save the plot
    fig4.tight_layout()
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.savefig('%spred_propose%d.png' 
        % (full_directory, i_trials + 1))
    if (i_trials == 0) or (((i_trials + 1) % SAVE_TRIALS) == 0):
        plt.savefig('%spred_propose%d.eps' 
            % (full_directory, i_trials + 1))

if __name__ == "__main__":
    main()