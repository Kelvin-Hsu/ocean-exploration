import numpy as np
import shelve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
import time

def main():

    main_directory = "../../../Results/scott-reef/"

    # Comparing different starting locations
    def compare_starting_locations():

        directory0 = main_directory + 'loc1_20150819_235313__t200_q100000_ts250_qs500_method_LDE_start377500_8440000_hsteps30_horizon5000/'
        directory1 = main_directory + 'loc2_20150815_221358__t200_q100000_ts250_qs500_method_LDE_start380000_8440000_hsteps30_horizon5000/'
        directory2 = main_directory + 'loc3_20150816_232942__t200_q100000_ts250_qs500_method_LDE_start375000_8445000_hsteps30_horizon5000/'
        directory3 = main_directory + 'loc4_20150817_214222__t200_q100000_ts250_qs500_method_LDE_start365000_8445000_hsteps30_horizon5000/'
        directory4 = main_directory + 'loc5_20150819_235323__t200_q100000_ts250_qs500_method_LDE_start380000_8446000_hsteps30_horizon5000/'

        data0 = obtain_data(directory0, {'index': 0, 'label': 'Starting Location 1', 'steps': 200})
        data1 = obtain_data(directory1, {'index': 1, 'label': 'Starting Location 2', 'steps': 200})
        data2 = obtain_data(directory2, {'index': 2, 'label': 'Starting Location 3', 'steps': 200})
        data3 = obtain_data(directory3, {'index': 3, 'label': 'Starting Location 4', 'steps': 200})
        data4 = obtain_data(directory3, {'index': 4, 'label': 'Starting Location 5', 'steps': 200})

        plot_data(main_directory, data0, data1, data2, data3, data4, ncolors = 5, descript = 'locations')
        logging.info('Compared starting locations')

    # Comparing different horizons
    def compare_horizons():

        directory0 = main_directory + 'h_compare_20150815_062728__t200_q100000_ts250_qs500_method_LDE_start375000.08440000.0_hsteps30_horizon5000.0/'
        directory1 = main_directory + 'h_compare_20150815_062732__t200_q100000_ts250_qs500_method_LDE_start375000.08440000.0_hsteps30_horizon7500.0/'
        directory2 = main_directory + 'h_compare_20150815_062834__t200_q100000_ts250_qs500_method_LDE_start375000.08440000.0_hsteps30_horizon6000.0/'

        data0 = obtain_data(directory0, {'index': 0, 'label': 'Horizon: 5000 m', 'steps': 200})
        data1 = obtain_data(directory1, {'index': 1, 'label': 'Horizon: 7500 m', 'steps': 200})
        data2 = obtain_data(directory2, {'index': 2, 'label': 'Horizon: 6000 m', 'steps': 200})

        plot_data(main_directory, data0, data1, data2, ncolors = 3, descript = 'horizons')
        logging.info('Compared horizons')

    # Compare with other methods
    def compare_methods():

        directory00 = main_directory + 'loc1_20150819_235313__t200_q100000_ts250_qs500_method_LDE_start377500_8440000_hsteps30_horizon5000/'
        directory01 = main_directory + 'loc2_20150815_221358__t200_q100000_ts250_qs500_method_LDE_start380000_8440000_hsteps30_horizon5000/'
        directory10 = main_directory + 'loc_20150816_015647__t200_q100000_ts250_qs500_method_MIE_GREEDY_start377500.08440000.0_hsteps30_horizon5000.0/'
        directory11 = main_directory + 'loc_20150816_015641__t200_q100000_ts250_qs500_method_MIE_GREEDY_start380000.08440000.0_hsteps30_horizon5000.0/'
        directory20 = main_directory + 'loc_20150816_064319__t200_q100000_ts250_qs500_method_RANDOM_start377500.08440000.0_hsteps30_horizon5000.0/'
        directory21 = main_directory + 'loc_20150816_081923__t200_q100000_ts250_qs500_method_RANDOM_start380000.08440000.0_hsteps30_horizon5000.0/'
        directory30 = main_directory + 'loc_20150821_022303__t200_q100000_ts250_qs500_method_MCJE_start377500.08440000.0_hsteps30_horizon5000.0/'
        directory31 = main_directory + 'loc_20150821_085829__t200_q100000_ts250_qs500_method_MCJE_start380000.08440000.0_hsteps30_horizon5000.0/'
        directory40 = main_directory + 'loc_20150818_085143__t200_q100000_ts250_qs500_method_MIE_start377500.08440000.0_hsteps30_horizon5000.0/'
        directory41 = main_directory + 'loc_20150818_221107__t200_q100000_ts250_qs500_method_MIE_start380000.08440000.0_hsteps30_horizon5000.0/'
        directory50 = main_directory + 'loc_20150822_141334__t200_q100000_ts250_qs500_method_FIXED_start377500.08440000.0_hsteps30_horizon5000.0/'
        directory51 = main_directory + 'loc_20150818_120403__t200_q100000_ts250_qs500_method_FIXED_start380000.08440000.0_hsteps30_horizon5000.0/'
        directory60 = main_directory + 'loc_20150819_095126__t200_q100000_ts250_qs500_method_FIXED_start377500.08440000.0_hsteps30_horizon5000.0/'
        directory61 = main_directory + 'loc_20150819_095211__t200_q100000_ts250_qs500_method_FIXED_start380000.08440000.0_hsteps30_horizon5000.0/'

        data00 = obtain_data(directory00, {'index': 0, 'label': 'Location 1 with LDE', 'steps': 200})
        data01 = obtain_data(directory01, {'index': 0, 'label': 'Location 2 with LDE', 'steps': 200})
        data10 = obtain_data(directory10, {'index': 1, 'label': 'Location 1 with GREEDY', 'steps': 200})
        data11 = obtain_data(directory11, {'index': 1, 'label': 'Location 2 with GREEDY', 'steps': 200})
        data20 = obtain_data(directory20, {'index': 2, 'label': 'Location 1 with RANDOM', 'steps': 200})
        data21 = obtain_data(directory21, {'index': 2, 'label': 'Location 2 with RANDOM', 'steps': 200})
        data30 = obtain_data(directory30, {'index': 3, 'label': 'Location 1 with MCJIE', 'steps': 200})
        data31 = obtain_data(directory31, {'index': 3, 'label': 'Location 2 with MCJIE', 'steps': 200})
        data40 = obtain_data(directory40, {'index': 4, 'label': 'Location 1 with MIE', 'steps': 200})
        data41 = obtain_data(directory41, {'index': 4, 'label': 'Location 2 with MIE', 'steps': 200})
        data50 = obtain_data(directory50, {'index': 5, 'label': 'Location 1 with FIXED - SPIRAL', 'steps': 200})
        data51 = obtain_data(directory51, {'index': 5, 'label': 'Location 2 with FIXED - SPIRAL', 'steps': 200})
        data60 = obtain_data(directory60, {'index': 6, 'label': 'Location 1 with FIXED - LINES', 'steps': 200})
        data61 = obtain_data(directory61, {'index': 6, 'label': 'Location 2 with FIXED - LINES', 'steps': 200})

        plot_data(main_directory, data00, data01, data10, data11, data20, data21, data30, data31, data40, data41, data50, data51, data60, data61, ncolors = 7, descript = 'methods', label_font_size = 18)
        logging.info('Compared methods')
        rank_data(data00, data01, data10, data11, data20, data21, data30, data31, data40, data41, data50, data51, data60, data61)

    logging.basicConfig(level = logging.DEBUG)

    compare_starting_locations()
    compare_horizons()
    compare_methods()

    plt.show()

def rank_data(*args):

    performances = np.array([arg[0][arg[-1].get('steps') - 1] for arg in args])
    names = [arg[-1].get('label') for arg in args]
    ind = performances.argsort()
    table = [(names[i], performances[i]) for i in ind]
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

def plot_data(directory, *args, ncolors = 1, descript = '', label_font_size = 24):

    L = 0.0
    colors = cm.rainbow(np.linspace(0 + L, 1 - L, num = ncolors))

    fontsize = 40
    axis_tick_font_size = 24
    
    params = {
        'backend': 'pdf',
        'axes.labelsize': 10,
        'text.fontsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        # 'text.usetex': True,
        'figure.figsize': fig_size(350.0)
    }

    plt.rc_context(params)

    fig = plt.figure(figsize = (20, 20))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    for arg in args:

        miss_ratio_array, yq_lde_mean_array, yq_mie_mean_array, info = arg

        iterations = np.arange(miss_ratio_array.shape[0]) + 1
        
        color = colors[info['index']]
        label = info['label']
        steps = info['steps']

        iterations_plt = np.append(0, iterations[:steps]) * (5000.0/30.0)
        miss_ratio_plt = np.append(41.54, 100 * miss_ratio_array[:steps])
        yq_lde_plt = np.append(-1.03, yq_lde_mean_array[:steps])
        yq_mie_plt = np.append(2.14, yq_mie_mean_array[:steps])

        ax1.plot(iterations_plt, miss_ratio_plt, c = color, label = label)
        ax1.set_ylim((0, 50))
        ax2.plot(iterations_plt, yq_lde_plt, c = color, label = label)
        ax3.plot(iterations_plt, yq_mie_plt, c = color, label = label)

    ax1.legend(bbox_to_anchor = (0., 0.0, 1., .05), loc = 3,
           ncol = 4, borderaxespad = 0., fontsize = label_font_size)

    plt.subplot(3, 1, 1)
    plt.title('Percentage of Prediction Misses', fontsize = fontsize)
    plt.ylabel('Misses (%)', fontsize = fontsize)
    plt.gca().set_xticklabels( () )

    plt.subplot(3, 1, 2)
    plt.title('Average Marginalised Linearised Differential Entropy', fontsize = fontsize)
    plt.ylabel('Entropy (nats)', fontsize = fontsize)
    plt.gca().set_xticklabels( () )

    plt.subplot(3, 1, 3)
    plt.title('Average Marginalised Information Entropy', fontsize = fontsize)
    plt.ylabel('Entropy (nats)', fontsize = fontsize)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    
    plt.gca().set_xlabel('Distance Traveled (m)', fontsize = fontsize)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    # Save the plot
    plt.tight_layout()
    plt.savefig('%scompare_%s.eps' % (directory, descript))

if __name__ == "__main__":
    main()