"""
Module  : Ocean Environment Modeling
File    : main.py

Author  : Kelvin Hsu
Date    : 1st July 2015
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from computers import gp
import computers.unsupervised.whitening as pre
import time
import logging
import sys
import os
import isea

def kerneldef(h, k):
    """Define the kernel used in the classifier"""
    return  h(1e-3, 1e3, 10)*k('gaussian', 
            [h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1), 
            h(1e-3, 1e3, 0.1), h(1e-3, 1e3, 0.1)])

def main():

    """ Test Options """
    SEED = 100

    """Model Options"""
    SAVE_RESULTS = True

    approxmethod = 'laplace'
    multimethod = 'OVA'
    fusemethod = 'EXCLUSION'
    responsename = 'probit'
    batchstart = True
    walltime = 3600.0
    train = True
    white_fn = pre.standardise

    n_train_sample = 300
    n_query_sample = 200000

    """Visualisation Options"""
    mycmap = cm.get_cmap(name = 'jet', lut = None)
    vis_fix_range = True
    vis_x_min = 360000
    vis_x_max = 390000
    vis_y_min = 8430000
    vis_y_max = 8450000
    vis_range = (vis_x_min, vis_x_max, vis_y_min, vis_y_max)

    """Initialise Result Logging"""
    if SAVE_RESULTS:
        home_directory = "../../../Results/ocean-exploration/"
        save_directory = "scott_reef_t%d_q%d/" % (n_train_sample, n_query_sample)

        full_directory = gp.classifier.utils.create_directories(save_directory, 
            home_directory = home_directory, append_time = True)

        textfilename = '%slog.txt' % full_directory

    """Logging Options"""
    logging.basicConfig(level = logging.DEBUG,
                        format =    '%(asctime)s %(name)-12s '\
                                    '%(levelname)-8s %(message)s',
                        datefmt = '%m-%d %H:%M',
                        filename = textfilename,
                        filemode = 'a')
    gp.classifier.set_multiclass_logging_level(logging.DEBUG)

    # Define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # Set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    # Tell the handler to use this format
    console.setFormatter(formatter)

    # Add the handler to the root logger
    logging.getLogger().addHandler(console)

    """Process Options"""
    model_options = {   'trainingsample': n_train_sample,
    					'querysample': n_query_sample,
    					'approxmethod': approxmethod,
                        'multimethod': multimethod,
                        'fusemethod': fusemethod,
                        'responsename': responsename,
                        'batchstart': batchstart,
                        'walltime': walltime,
                        'train': train}

    logging.info(model_options)

    """File Locations"""
    directory_data = '../../../Data/'
    filename_training_data = 'training_data_unmerged.npz'
    filename_query_points = 'query_points.npz'

    """Load Data"""
    (X_all, F_all, y_all, Xq_all, Fq_all) = isea.utils.load(directory_data, 
        filename_training_data, filename_query_points)

    n_train = F_all.shape[0]
    n_query = Fq_all.shape[0]
    k_features = F_all.shape[1]
    assert k_features == 5
    feature_names = [   'Bathymetry (Depth)', 
                        'Aspect (Short Scale)',
                        'Rugosity (Short Scale)',
                        'Aspect (Long Scale)',
                        'Rugosity (Long Scale)']

    logging.info('Raw Number of Training Points: %d' % n_train)
    logging.info('Raw Number of Query Points: %d' % n_query)

    """Sample Training Data and Query Points"""
    np.random.seed(SEED)
    (X_sample, F_sample, y_sample, Xq_sample, Fq_sample) \
        = isea.utils.sample(X_all, F_all, y_all, Xq_all, Fq_all, 
            n_train_sample = n_train_sample, n_query_sample = n_query_sample)

    y_unique_sample = np.unique(y_sample)
    assert y_unique_sample.shape[0] == 17
    logging.info('There are %d unique labels' % y_unique_sample.shape[0])

    """Whiten the feature space"""
    logging.info('Applying whitening on training and query features...')
    feature_fn = isea.compose_white_feature_fn(Xq_sample, Fq_sample, white_fn)
    Fw_sample, white_params = white_fn(F_sample)
    Fqw_sample = white_fn(Fq_sample, params = white_params)

    logging.info('Whitening Parameters:')
    logging.info(white_params)

    """Visualise Sampled Training Locations"""
    fig = plt.figure(figsize = (19.2, 10.8))
    plt.scatter(
        X_sample[:, 0], X_sample[:, 1], 
        marker = 'x', c = y_sample, 
        vmin = y_unique_sample[0], vmax = y_unique_sample[-1], 
        cmap = mycmap)
    isea.utils.describe_plot(title = 'Training Labels', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Habitat Labels', cticks = y_unique_sample,
        vis_range = vis_range, aspect_equal = True)

    """Visualise Features at Sampled Query Locations"""
    for k in range(k_features):
        fig = plt.figure(figsize = (19.2, 10.8))
        isea.utils.scatter(
            Xq_sample[:, 0], Xq_sample[:, 1], 
            marker = 'x', c = Fq_sample[:, k], s = 5, 
            cmap = mycmap, colorcenter = True)
        isea.utils.describe_plot(clabel = '%s (Raw)' % feature_names[k])
        isea.utils.scatter(
            Xq_sample[:, 0], Xq_sample[:, 1], 
            marker = 'x', c = Fqw_sample[:, k], s = 5,
            cmap = mycmap, colorcenter = True)
        isea.utils.describe_plot(
            title = 'Feature: %s at Query Points' % feature_names[k], 
            xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
            clabel = '%s (Whitened)' % feature_names[k],
            vis_range = vis_range, aspect_equal = True)

    """Classifier Training"""
    logging.info('===Begin Classifier Training===')
    logging.info('Number of training points: %d' % n_train_sample)

    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # User can choose to batch start each binary classifier with different
    # initial hyperparameters for faster training
    if batchstart:
        initial_hyperparams = \
                                [    [225.01828874969513, 1.2618738490059052, 17.613593073339857, 12.679685345492906, 8.6262994637623169, 14.136073532069654], \
                                     [2.589083342814575, 1.1052301403959857, 4.3644907968320794, 4.0156800544013631, 1.9007076030011256, 3.9503021025566549], \
                                     [6.704649924638316, 7.5142456145585852, 10.182539495816165, 6.5811182358089964, 9.1368011211462683, 9.7400903506316272], \
                                     [2.1521775248096446, 1.5480845798959013, 1.7727587298253529, 1.9531407045936808, 2.6291143996909319, 3.0014523367867647], \
                                     [4.4145724529148573, 2.3177583295784854, 3.7037953308551175, 3.7294392645432604, 2.0975271260436825, 3.9495445442492572], \
                                     [2.2789691898199238, 1.3116551225946529, 4.363909816990545, 3.3402865492153566, 1.1820443956790891, 3.464914967442942], \
                                     [2.2351746154111178, 0.88313353460040367, 1.2254449251993957, 2.0047492138759808, 0.7359745470808734, 1.3441751232794052], \
                                     [2.7192795673173125, 1.9773387942329865, 1.4579897229047554, 1.9817457382237416, 1.5572169782422107, 1.8730303409358018], \
                                     [2.0758288682909476, 1.2612996582885267, 2.3759661774412209, 2.4093161358141555, 1.7870648339548159, 2.6745811799029862], \
                                     [11.129910487357249, 1.5379441304334878, 4.5627982066806041, 6.5643558329665437, 2.2466955217967244, 6.6273259324841334], \
                                     [2.0279161647863813, 1.9089318650401494, 1.6193561676805104, 2.5866394113420017, 1.3784102883153633, 3.1982719173007177], \
                                     [3.3989164525541358, 1.7382334518504794, 1.8651494130520445, 2.2399330663269525, 3.4853142140708218, 3.2134802913286395], \
                                     [2.4642534973485368, 3.4337999718161663, 3.9824996504195851, 2.0670160184395363, 1.7337535572033278, 2.2896179419157208], \
                                     [12.88118605355896, 1.364920676715099, 12.017531667114847, 8.4028421151450079, 3.8158770382163691, 7.8630080376372842], \
                                     [170.14744754307256, 1.5154971208609203, 11.612388859494869, 14.390128191785918, 9.8434640241485472, 3.4595594042354083], \
                                     [136.11476775141347, 4.5649178398095103, 17.054161938288676, 43.509295473832708, 36.843822718649378, 4.235728726826542], \
                                     [5.1892256356491009, 4.0010875696881456, 6.7249839105493585, 4.0905271432989778, 5.8275763934819604, 5.0612290664566961] ]
        # SEED 100 NTRAIN 300
        initial_hyperparams = \
                                [    [362.92591030813242, 1.2151750269695372, 59.566522727138313, 27.215097108437007, 4.2668780608419823, 48.646891241197267], \
                                     [28.410909586471902, 1.6547409996041162, 35.181731533678523, 54.811971329968053, 6.7981974974178918, 37.064340344985489], \
                                     [7.6216625372646538, 90.058784012697217, 116.66482695523602, 55.535503470015364, 110.92321547392022, 77.071141621387568], \
                                     [1.5768029665325165, 3.8878145465622507, 24.008839080730912, 33.765958354615393, 23.123914164426122, 2.8079270862221195], \
                                     [2.2046531818048596, 1.8240437278704882, 20.860960465057293, 22.080682780942031, 28.487659895152998, 19.796263644578165], \
                                     [2.5858331838159381, 5.1532258558761175, 61.770088965563701, 18.340405139610926, 31.973013428711642, 17.109458612526272], \
                                     [2.1434241790231678, 1.3242146172720932, 30.07972956064993, 24.787134447383387, 9.0122488386123223, 14.774183226305345], \
                                     [4.4841194248268836, 14.257195977751005, 31.238101053548693, 26.985981259344396, 29.671065405379018, 2.5084814457940734], \
                                     [9.7302797822839633, 2.8285864857988483, 22.609715889910294, 22.219802976467555, 2.3881420894826375, 14.421650197536261], \
                                     [12.829916611586468, 1.3888575814207087, 27.64578091319358, 13.982368669512779, 2.4597498808197567, 26.057722893351723], \
                                     [2.2665953474458513, 14.592612185574257, 19.975325070791744, 3.1283680590160254, 8.8368591372471847, 4.8835669147886787], \
                                     [157.78294780537757, 4.0960242089181298, 5.274239168403998, 12.521494925287476, 2.4605856762532912, 7.7272167882604688], \
                                     [3.4004984263466644, 2.4408967175913712, 39.160176505939809, 27.967029108997064, 11.057618967917747, 15.585399892760641], \
                                     [42.682242951397328, 1.9009329202410523, 68.066917448768791, 60.666463073240692, 8.9789229381446543, 45.757254434068436], \
                                     [495.14458069351053, 1.4230543562260116, 4.5835141350525692, 34.794823077448399, 12.908805452331627, 15.532098734787946], \
                                     [11.862345426556194, 4.4042160200831679, 29.712163671721687, 41.345655550097156, 58.247829889166795, 17.72698722003086], \
                                     [7.3751943783426457, 40.453604512432129, 47.516981955027035, 45.2179252285366, 35.365341121902262, 82.502067880958208] ]
        batch_config = gp.batch_start(optimiser_config, initial_hyperparams)
        logging.info('Using Batch Start Configuration')
        
    else:
        batch_config = optimiser_config
    
    # Obtain the response function
    responsefunction = gp.classifier.responses.get(responsename)

    # Train the classifier!
    logging.info('Learning...')
    learned_classifier = gp.classifier.learn(Fw_sample, y_sample, 
        kerneldef, responsefunction, batch_config, 
        multimethod = multimethod, approxmethod = approxmethod, 
        train = train, ftol = 1e-10)

    # Print the learnt kernel with its hyperparameters
    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(print_function, 
        learned_classifier, y_unique_sample)

    # Print the matrix of learned classifier hyperparameters
    logging.info('Matrix of learned hyperparameters')
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)

    """Classifier Prediction"""

    yq_pred, yq_mie, yq_lde = predictions(learned_classifier, Fqw)

    """Visualise Query Prediction and Entropy"""
    fig = plt.figure(figsize = (19.2, 10.8))
    isea.utils.scatter(
        Xq_sample[:, 0], Xq_sample[:, 1], 
        marker = 'x', c = yq_pred, s = 5, 
        vmin = y_unique_sample[0], vmax = y_unique_sample[-1], 
        cmap = mycmap, colorcenter = True)
    isea.utils.describe_plot(title = 'Query Predictions', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Habitat Labels', cticks = y_unique_sample,
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    isea.utils.scatter(
        Xq_sample[:, 0], Xq_sample[:, 1], 
        marker = 'x', c = yq_mie, s = 5, cmap = cm.coolwarm, 
        colorcenter = True)
    isea.utils.describe_plot(title = 'Query Information Entropy', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Information Entropy',
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    isea.utils.scatter(
        Xq_sample[:, 0], Xq_sample[:, 1], 
        marker = 'x', c = np.log(yq_mie), s = 5, 
        cmap = cm.coolwarm, colorcenter = True)
    isea.utils.describe_plot(title = 'Log Query Information Entropy', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Information Entropy',
        vis_range = vis_range, aspect_equal = True)

    fig = plt.figure(figsize = (19.2, 10.8))
    isea.utils.scatter(
        Xq_sample[:, 0], Xq_sample[:, 1], 
        marker = 'x', c = yq_lde, s = 5, 
        cmap = cm.coolwarm, colorcenter = True)
    isea.utils.describe_plot(title = 'Query Linearised Differential Entropy', 
        xlabel = 'x [Eastings (m)]', ylabel = 'y [Northings (m)]', 
        clabel = 'Differential Entropy',
        vis_range = vis_range, aspect_equal = True)

    """Visualise Query Draws"""

    if SAVE_RESULTS:
        gp.classifier.utils.save_all_figures(full_directory)

    plt.show()

def predictions(learned_classifier, Fqw):

    logging.info('Caching Predictor...')
    predictor = gp.classifier.query(learned_classifier, Fqw)
    logging.info('Computing Expectance...')
    fq_exp = gp.classifier.expectance(learned_classifier, predictor)
    logging.info('Computing Variance...')
    fq_var = gp.classifier.variance(learned_classifier, predictor)
    logging.info('Computing Prediction Probabilities...')
    yq_prob = gp.classifier.predict_from_latent(fq_exp, fq_var, 
        learned_classifier, fusemethod = fusemethod)
    logging.info('Computing Prediction...')
    yq_pred = gp.classifier.classify(yq_prob, y_unique_sample)
    logging.info('Computing Prediction Information Entropy...')
    yq_mie = gp.classifier.entropy(yq_prob)    
    logging.info('Computing Linearised Differential Entropy...')
    yq_lde = gp.classifier.linearised_entropy(fq_exp, fq_var, 
        learned_classifier)
    return yq_pred, yq_mie, yq_lde

if __name__ == "__main__":
    main()
