import numpy as np
import shelve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
import time
import matplotlib.ticker as ticker

def main():

    main_directory = "../../../Results/scott-reef/"

    # Compare with other methods
    def compare_methods():

        directory0 = main_directory + 'missions_new_20151002_090448__t17_q100000_ts250_qs500_method_LMDE_start375000.08440000.0_hsteps30_horizon5000.0/'
        directory1 = main_directory + 'missions_new_20151002_145418__t17_q100000_ts250_qs500_method_MCPIE_start375000.08440000.0_hsteps30_horizon5000.0/'
        directory2 = main_directory + 'missions_new_20151004_051911__t17_q100000_ts250_qs500_method_AMPIE_start375000.08440000.0_hsteps30_horizon5000.0/'
        directory3 = main_directory + 'missions_new_20151003_010129__t17_q100000_ts250_qs500_method_AMPIE_GREEDY_start375000.08440000.0_hsteps30_horizon5000.0/'
        directory4 = main_directory + 'missions_new_20151003_082218__t17_q100000_ts250_qs500_method_RANDOM_start375000.08440000.0_hsteps30_horizon5000.0/'
        directory5 = main_directory + 'missions_new_20151003_131955__t17_q100000_ts250_qs500_method_FIXED_FTYPE_lines_start375000.08440000.0_hsteps30_horizon5000.0/'
        directory6 = main_directory + 'missions_new_20151003_234114__t17_q100000_ts250_qs500_method_FIXED_FTYPE_spiral_start375000.08440000.0_hsteps30_horizon5000.0/'

        data0 = obtain_data(directory0, {'index': 0, 'label': 'LMDE Acquisition', 'steps': 200})
        data1 = obtain_data(directory1, {'index': 1, 'label': 'MCPIE Acquisition', 'steps': 200})
        data2 = obtain_data(directory2, {'index': 2, 'label': 'AMPIE Acquisition', 'steps': 200})
        data3 = obtain_data(directory3, {'index': 3, 'label': 'GREEDY-PIE Acquisition', 'steps': 200})
        data4 = obtain_data(directory4, {'index': 4, 'label': 'RANDOM Exploration', 'steps': 200})
        data5 = obtain_data(directory5, {'index': 5, 'label': 'LINES Exploration', 'steps': 200})
        data6 = obtain_data(directory6, {'index': 6, 'label': 'SPIRAL Exploration', 'steps': 200})

        plot_data(main_directory, data0, data1, data2, data3, data4, data5, data6, ncolors = 7, descript = 'methods', label_font_size = 18)

        logging.info('Compared methods')
        rank_data(data0, data1, data2, data3, data4, data5, data6)

    logging.basicConfig(level = logging.DEBUG)

    compare_methods()

    plt.show()

def rank_data(*args):

    performances = np.array([arg[0][arg[-1].get('steps') - 1] for arg in args])
    names = [arg[-1].get('label') for arg in args]
    ind = performances.argsort()
    table = [(names[i], np.round(100 * performances[i], 2)) for i in ind]
    [print(t) for t in table]
    print('----------')
    [print(t) for t in table if 'Location 1' in t[0]]
    print('----------')
    [print(t) for t in table if 'Location 2' in t[0]]

def obtain_data(directory, info):

    try:
        history = np.load('%shistory.npz' % directory)
        miss_ratio_array = history['miss_ratio_array']
        yq_lde_mean_array = history['yq_lde_mean_array']
        yq_mie_mean_array = history['yq_mie_mean_array']
        logging.info('Obtained data for {0}'.format(info))
    except:
        miss_ratio_array = np.nan * np.ones(info['steps'])
        yq_lde_mean_array = np.nan * np.ones(info['steps'])
        yq_mie_mean_array = np.nan * np.ones(info['steps'])
        logging.info('Failed to obtain data for {0}'.format(info))
    return miss_ratio_array, yq_lde_mean_array, yq_mie_mean_array, info

def fig_size(fig_width_pt):
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt# width in inches
    fig_height = fig_width * golden_mean    # height in inches
    return fig_width, fig_height

def plot_data(directory, *args, ncolors = 1, descript = '', label_font_size = 24, ncol = 4):

    L = 0.0
    colors = cm.rainbow(np.linspace(0 + L, 1 - L, num = ncolors))

    fontsize = 50
    axis_tick_font_size = 24
    
    params = {
        'backend': 'ps',
        # 'axes.labelsize': 10,
        # 'text.fontsize': 10,
        # 'legend.fontsize': 10,
        # 'xtick.labelsize': 8,
        # 'ytick.labelsize': 8,
        'text.usetex': True,
        'figure.figsize': fig_size(350.0)
    }

    plt.rc_context(params)

    fig = plt.figure(figsize = (20, 15))
    ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(212)

    for arg in args:

        miss_ratio_array, yq_lde_mean_array, yq_mie_mean_array, info = arg

        iterations = np.arange(miss_ratio_array.shape[0]) + 1
        
        color = colors[info['index']]
        label = info['label']
        steps = info['steps']
        if 'linestyle' in info:
            linestyle = info['linestyle']
        else:
            linestyle = 'solid'

        iterations_plt = np.append(0, iterations[:steps]) * (5000.0/30.0)
        miss_ratio_plt = np.append(99.39, 100 * miss_ratio_array[:steps])
        yq_lde_plt = np.append(-0.20, yq_lde_mean_array[:steps])
        yq_mie_plt = np.append(2.56, yq_mie_mean_array[:steps])

        ax1.plot(iterations_plt, miss_ratio_plt, c = color, label = label, linewidth = 2.0, linestyle = linestyle)
        ax1.set_ylim((0, 100))
        # ax2.plot(iterations_plt, yq_lde_plt, c = color, label = label, linewidth = 2.0, linestyle = linestyle)
        ax3.plot(iterations_plt, yq_mie_plt, c = color, label = label, linewidth = 2.0, linestyle = linestyle)

    ax1.legend(bbox_to_anchor = (0., 0.0, 1., .05), loc = 3,
           ncol = ncol, borderaxespad = 0., fontsize = label_font_size)

    ax1.set_title('Percentage of Map Prediction Misses', fontsize = fontsize)
    ax1.set_ylabel('Misses (\%)', fontsize = fontsize)
    ax1.set_xticklabels( () )

    # ax2.set_title('Average Marginalised L. Model Differential Entropy', fontsize = fontsize)
    # ax2.set_ylabel('Entropy (nats)', fontsize = fontsize)
    # ax2.set_xticklabels( () )

    ax3.set_title('Average Marginalised Prediction Information Entropy', fontsize = fontsize)
    ax3.set_ylabel('Entropy (nats)', fontsize = fontsize)
    ax3.get_xaxis().get_major_formatter().set_useOffset(False)
    ax3.set_xlabel('Distance Traveled (km)', fontsize = fontsize)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size)
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(axis_tick_font_size)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size)

    # ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e3))
    # ax3.xaxis.set_major_formatter(ticks)
    # ax3.yaxis.set_major_formatter(ticks)

    # Save the plot
    fig.tight_layout()
    fig.savefig('%sserial_compare_%s.eps' % (directory, descript))

if __name__ == "__main__":
    main()