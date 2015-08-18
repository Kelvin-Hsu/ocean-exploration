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

        directory0 = main_directory + ''
        directory1 = main_directory + ''
        directory2 = main_directory + ''
        directory3 = main_directory + ''

        data0 = obtain_data(directory0, {'index': 0, 'label': 'Starting Location 1', 'steps': 200})
        data1 = obtain_data(directory1, {'index': 1, 'label': 'Starting Location 2', 'steps': 200})
        data2 = obtain_data(directory2, {'index': 2, 'label': 'Starting Location 3', 'steps': 200})
        data3 = obtain_data(directory3, {'index': 3, 'label': 'Starting Location 4', 'steps': 200})

        plot_data(main_directory, data0, data1, data2, data3, ncolors = 4)


    # Comparing different horizons
    def compare_horizons():

        directory0 = main_directory + 'h_compare_20150815_062728__t200_q100000_ts250_qs500_method_LDE_start375000.08440000.0_hsteps30_horizon5000.0'
        directory1 = main_directory + 'h_compare_20150815_062732__t200_q100000_ts250_qs500_method_LDE_start375000.08440000.0_hsteps30_horizon7500.0'
        directory2 = main_directory + 'h_compare_20150815_062834__t200_q100000_ts250_qs500_method_LDE_start375000.08440000.0_hsteps30_horizon6000.0'

        data0 = obtain_data(directory0, {'index': 0, 'label': 'Horizon: 5000 m', 'steps': 200})
        data1 = obtain_data(directory1, {'index': 1, 'label': 'Horizon: 7500 m', 'steps': 200})
        data2 = obtain_data(directory2, {'index': 2, 'label': 'Horizon: 6000 m', 'steps': 200})

        plot_data(main_directory, data0, data1, data2, ncolors = 3)

    # Compare with other methods
    def compare_methods():

        directory00 = main_directory + ''
        directory01 = main_directory + ''
        directory10 = main_directory + 'loc_20150816_015647__t200_q100000_ts250_qs500_method_MIE_GREEDY_start377500.08440000.0_hsteps30_horizon5000.0'
        directory11 = main_directory + 'loc_20150816_015641__t200_q100000_ts250_qs500_method_MIE_GREEDY_start380000.08440000.0_hsteps30_horizon5000.0'
        directory20 = main_directory + 'loc_20150816_064319__t200_q100000_ts250_qs500_method_RANDOM_start377500.08440000.0_hsteps30_horizon5000.0'
        directory21 = main_directory + 'loc_20150816_081923__t200_q100000_ts250_qs500_method_RANDOM_start380000.08440000.0_hsteps30_horizon5000.0'
        directory30 = main_directory + 'loc_20150816_202553__t200_q100000_ts250_qs500_method_MCJE_start377500.08440000.0_hsteps30_horizon5000.0'
        directory31 = main_directory + 'loc_20150816_202603__t200_q100000_ts250_qs500_method_MCJE_start380000.08440000.0_hsteps30_horizon5000.0'
        directory40 = main_directory + 'loc_20150816_011248__t200_q100000_ts250_qs500_method_MIE_start377500.08440000.0_hsteps30_horizon5000.0'
        directory41 = main_directory + 'loc_20150816_002100__t200_q100000_ts250_qs500_method_MIE_start380000.08440000.0_hsteps30_horizon5000.0'

        data00 = obtain_data(directory00, {'index': 0, 'label': 'Location 1 with LDE', 'steps': 200})
        data01 = obtain_data(directory01, {'index': 0, 'label': 'Location 2 with LDE', 'steps': 200})
        data10 = obtain_data(directory10, {'index': 1, 'label': 'Location 1 with GREEDY', 'steps': 200})
        data11 = obtain_data(directory11, {'index': 1, 'label': 'Location 2 with GREEDY', 'steps': 200})
        data20 = obtain_data(directory20, {'index': 2, 'label': 'Location 1 with OPEN LOOP', 'steps': 200})
        data21 = obtain_data(directory21, {'index': 2, 'label': 'Location 2 with OPEN LOOP', 'steps': 200})
        data30 = obtain_data(directory30, {'index': 3, 'label': 'Location 1 with MCJIE', 'steps': 140})
        data31 = obtain_data(directory31, {'index': 3, 'label': 'Location 2 with MCJIE LOOP', 'steps': 140})
        data40 = obtain_data(directory40, {'index': 4, 'label': 'Location 1 with MIE', 'steps': 9})
        data41 = obtain_data(directory41, {'index': 4, 'label': 'Location 2 with MIE', 'steps': 38})

        plot_data(main_directory, data00, data01, data10, data11, data20, data21, data30, data31, data40, data41, ncolors = 5)

    compare_starting_locations()
    compare_horizons()
    compare_methods()
    
    plt.show()

def obtain_data(directory, info):

    history = np.load('%shistory.npz' % directory)

    miss_ratio_array = history['miss_ratio_array']
    yq_lde_mean_array = history['yq_lde_mean_array']
    yq_mie_mean_array = history['yq_mie_mean_array']

    return miss_ratio_array, yq_lde_mean_array, yq_mie_mean_array, info

def plot_data(directory, *args, ncolors = 1):

    fig = plt.figure(figsize = (20, 20))

    L = 0.2
    colors = cm.rainbow(np.linspace(0 + L, 1 - L, num = ncolors))

    fontsize = 24
    axis_tick_font_size = 14

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    for arg in args:

        miss_ratio_array, yq_lde_mean_array, yq_mie_mean_array, info = arg

        steps = np.arange(mistake_ratio_array.shape[0]) + 1
        
        color = colors[info['index']]
        label = info['label']
        steps = info['steps']

        ax1.plot(steps, 100 * miss_ratio_array[:steps], c = color, label = label)
        ax2.plot(steps, yq_lde_mean_array[:steps], c = color, label = label)
        ax3.plot(steps, yq_mie_mean_array[:steps], c = color, label = label)

    ax1.legend(bbox_to_anchor=(0., 0.8, 1., .05), loc=3,
           ncol=4, borderaxespad=0.)

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
    
    plt.gca().set_xlabel('Steps', fontsize = fontsize)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(axis_tick_font_size) 

    # Save the plot
    plt.tight_layout()
    plt.savefig('%shistory.png' % directory)

if __name__ == "__main__":
    main()