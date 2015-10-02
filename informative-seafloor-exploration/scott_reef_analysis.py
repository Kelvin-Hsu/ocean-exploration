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
import sea
import shutil

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

    """Test Options"""
    FILENAME = 'scott_reef_analysis.py'
    T_SEED = sea.io.parse('-tseed', 250)
    Q_SEED = sea.io.parse('-qseed', 500) 
    N_TRAIN = sea.io.parse('-ntrain', 200)
    N_QUERY = sea.io.parse('-nquery', 100000)
    NOTRAIN = sea.io.parse('-skiptrain', False)
    MODEL_ONLY = sea.io.parse('-model-only', False)
    LONG_SCALE_ONLY = sea.io.parse('-long-scale', False)
    BATCH_START = sea.io.parse('-batch-start', 'on')

    MISSION_LENGTH = sea.io.parse('-mission-length', 0)
    METHOD = sea.io.parse('-method', 'LMDE')
    GREEDY = sea.io.parse('-greedy', False)
    N_TRIALS = sea.io.parse('-ntrials', 200)
    START_POINT1 = sea.io.parse('-start', 375000.0, arg = 1)
    START_POINT2 = sea.io.parse('-start', 8440000.0, arg = 2)
    H_STEPS = sea.io.parse('-hsteps', 30)
    HORIZON = sea.io.parse('-horizon', 5000.0)
    CHAOS = sea.io.parse('-chaos', False)
    M_STEP = sea.io.parse('-mstep', 1)
    N_DRAWS = sea.io.parse('-ndraws', 500)
    DEPTH_PENALTY = sea.io.parse('-depth-penalty', False)
    SKIP_FEATURE_PLOT = sea.io.parse('-skip-feature-plot', False)
    TWO_COLORBAR = sea.io.parse('-two-colorbar', False)

    FONTSIZE = 50
    FONTNAME = 'Sans Serif'
    TICKSIZE = 24
    SAVE_TRIALS = 25

    """Model Options"""
    SAVE_RESULTS = True

    approxmethod = 'laplace'
    multimethod = 'OVA'
    fusemethod = 'EXCLUSION'
    responsename = 'probit'
    batchstart = True if (BATCH_START == 'on') else False
    batchlearn = False
    walltime = 3600.0
    train = not NOTRAIN
    white_fn = pre.standardise

    n_train = N_TRAIN
    n_query = N_QUERY

    """Visualisation Options"""
    mycmap = cm.get_cmap(name = 'jet', lut = None)
    vis_fix_range = True
    vis_x_min = 360000
    vis_x_max = 390000
    vis_y_min = 8430000
    vis_y_max = 8450000
    vis_range = (vis_x_min, vis_x_max, vis_y_min, vis_y_max)
    colorcenter_analysis = 'mean'
    colorcenter_lde = colorcenter_analysis
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

    assert len(y_names_all) == 25
               
    rcparams = {
        'backend': 'pdf',
        'axes.labelsize': TICKSIZE,
        'text.fontsize': FONTSIZE,
        'legend.fontsize': FONTSIZE,
        'xtick.labelsize': TICKSIZE,
        'ytick.labelsize': TICKSIZE,
        'text.usetex': True,
        'figure.figsize': sea.vis.fig_size(350.0)
    }

    plt.rc_context(rcparams)
    # map_kwargs = {'alpha': 0.5, 'edgecolors': 'none', 's': 15}
    map_kwargs = {'marker': 'x', 's': 5}

    """Initialise Result Logging"""
    if SAVE_RESULTS:
        home_directory = "../../../Results/scott-reef/"
        save_directory = "t%d_q%d_ts%d_qs%d_method_%s%s_start%.1f%.1f_"\
        "hsteps%d_horizon%.1f/" % (N_TRAIN, N_QUERY, T_SEED, Q_SEED, 
                METHOD, '_GREEDY' if GREEDY else '', 
                START_POINT1, START_POINT2, H_STEPS, HORIZON)
        full_directory = gp.classifier.utils.create_directories(
            save_directory, 
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
    test_options  = {   'T_SEED': T_SEED,
                        'Q_SEED': Q_SEED,
                        'N_TRAIN': N_TRAIN,
                        'N_QUERY': N_QUERY,
                        'NOTRAIN': NOTRAIN,
                        'MODEL_ONLY': MODEL_ONLY,
                        'LONG_SCALE_ONLY': LONG_SCALE_ONLY,
                        'METHOD': METHOD,
                        'GREEDY': GREEDY,
                        'N_TRIALS': N_TRIALS,
                        'START_POINT1': START_POINT1,
                        'START_POINT2': START_POINT2,
                        'H_STEPS': H_STEPS,
                        'HORIZON': HORIZON,
                        'CHAOS': CHAOS,
                        'M_STEP': M_STEP,
                        'N_DRAWS': N_DRAWS}

    model_options = {   'approxmethod': approxmethod,
                        'multimethod': multimethod,
                        'fusemethod': fusemethod,
                        'responsename': responsename,
                        'batchstart': batchstart,
                        'batchlearn': batchlearn,
                        'walltime': walltime,
                        'train': train}

    logging.info(sys.argv)
    logging.info(test_options)
    logging.info(model_options)
    
    """File Locations"""
    directory_data = '../../../Data/'
    filename_training_data = 'training_data_unmerged.npz'
    filename_query_points = 'query_points.npz'
    filename_truth = directory_data + 'truthmodel_t800_q100000_ts250_qs500.npz'
    filename_start = directory_data + 'finalmodel_t200_q100000_ts250_qs500'\
        '_method_LMDE_start377500_8440000_hsteps30_horizon5000.npz'

    """Sample Training Data and Query Points"""
    if LONG_SCALE_ONLY:
        i_features = [0, 3, 4]
        feature_names = [   'Bathymetry (Depth)', 
                            'Aspect (Long Scale)',
                            'Rugosity (Long Scale)']
        kerneldef = kerneldef3
    else:
        i_features = [0, 1, 2, 3, 4]
        feature_names = [   'Bathymetry (Depth)', 
                            'Aspect (Short Scale)',
                            'Rugosity (Short Scale)',
                            'Aspect (Long Scale)',
                            'Rugosity (Long Scale)']
        kerneldef = kerneldef5

    X, F, y, Xq, Fq, i_train, i_query = \
        sea.io.sample(*sea.io.load(directory_data, 
            filename_training_data, filename_query_points), 
            n_train = n_train, n_query = n_query,
            t_seed = T_SEED, q_seed = Q_SEED, 
            features = i_features, unique_labels = True)

    start_indices = np.random.choice(np.arange(Xq.shape[0]), 
                            size = 2500, replace = False)

    yq_truth = sea.io.load_ground_truth(filename_truth, 
        assert_query_seed = Q_SEED)

    y_unique = np.unique(y)
    assert y_unique.shape[0] == 17
    logging.info('There are %d unique labels' % y_unique.shape[0])

    y_names = [y_names_all[i] for i in y_unique.astype(int)]
    logging.info('Habitat Labels: {0}'.format(y_names))

    """Whiten the feature space"""
    logging.info('Applying whitening on training and query features...')
    feature_fn = sea.feature.compose(Xq, Fq, white_fn)
    Fw, white_params = white_fn(F)
    Fqw = white_fn(Fq, params = white_params)

    k_features = F.shape[1]


    logging.info('Whitening Parameters:')
    logging.info(white_params)

    if not SKIP_FEATURE_PLOT:

        """Visualise Sampled Training Locations"""
        fig = plt.figure(figsize = (19.2, 10.8))
        plt.scatter(
            X[:, 0], X[:, 1], 
            marker = 'x', c = y, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)
        sea.vis.describe_plot(title = '(a) Training Labels', 
            xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
            clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
            vis_range = vis_range, aspect_equal = True, 
            fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, axis_scale = 1e3)
        if TWO_COLORBAR:
            plt.scatter(
                X[:, 0], X[:, 1], 
                marker = 'x', c = y, 
                vmin = y_unique[0], vmax = y_unique[-1], 
                cmap = mycmap)
            sea.vis.describe_plot(title = '(a) Training Labels', 
                xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
                clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_unique,
                vis_range = vis_range, aspect_equal = True, 
                fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, axis_scale = 1e3)
        fig.tight_layout()

        """Visualise Features at Sampled Query Locations"""
        letters = ['b', 'c', 'd', 'e', 'f']
        feature_labels = ['Depth', 'Aspect', 'Rugosity', 'Aspect', 'Rugosity']
        feature_units = ['$\mathrm{m}$', '$\mathrm{m}$/$\mathrm{m}$', '$\mathrm{m}^{2}$/$\mathrm{m}^{2}$', '$\mathrm{m}$/$\mathrm{m}$', '$\mathrm{m}^{2}$/$\mathrm{m}^{2}$']
        for k in range(k_features):
            fig = plt.figure(figsize = (19.2, 10.8))
            sea.vis.scatter(
                Xq[:, 0], Xq[:, 1], 
                c = Fq[:, k], colorcenter = 'mean', cmap = mycmap, **map_kwargs)
            sea.vis.describe_plot(
                title = '(%s) Feature: %s' % (letters[k], feature_names[k]), 
                xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
                clabel = '%s (%s)' % (feature_labels[k], feature_units[k]),
                vis_range = vis_range, aspect_equal = True, 
                fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, axis_scale = 1e3)
            if TWO_COLORBAR:
                sea.vis.scatter(
                    Xq[:, 0], Xq[:, 1], 
                    c = Fqw[:, k], colorcenter = 'mean', cmap = mycmap, **map_kwargs)
                sea.vis.describe_plot(
                    title = '(%s) Feature: %s' % (letters[k], feature_names[k]),
                    xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
                    clabel = 'Whitened %s' % feature_labels[k],
                    vis_range = vis_range, aspect_equal = True, 
                    fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, axis_scale = 1e3)
            fig.tight_layout()
            logging.info('Plotted feature map for: %s' % feature_names[k])

    """Classifier Training"""
    logging.info('===Begin Classifier Training===')
    logging.info('Number of training points: %d' % n_train)

    optimiser_config = gp.OptConfig()
    optimiser_config.sigma = gp.auto_range(kerneldef)
    optimiser_config.walltime = walltime

    # User can choose to batch start each binary classifier with different
    # initial hyperparameters for faster training
    if batchstart:

        if LONG_SCALE_ONLY:
            initial_hyperparams = \
                                    [    [138.04629034049745, 1.3026163180402612, 2.0609539037086777, 13.662777662608923], \
                                         [22.688174688988205, 0.85939961064783144, 3.1849018302605145, 16.259612437640932], \
                                         [4.9644266771973573, 12.863027059688255, 9.6883729945945429, 11.298947254002099], \
                                         [2.8722041026032517, 1.7551920752264887, 3.2957930714865951, 2.0261411106598501], \
                                         [1.95985751313864, 1.7015679970507851, 3.8989285686128072, 3.0227232782849662], \
                                         [4.5387717477104914, 6.1923651207093151, 2.6865876299919011, 1.2053732599769598], \
                                         [1.4406689190161075, 0.84772066855527117, 1.5917366549768182, 2.2383935939629236], \
                                         [1.5227503435926373, 2.3834299628669449, 1.8308476182313158, 1.3574417717717639], \
                                         [2.6346222397798864, 1.5157083797833799, 3.584415559552045, 3.6937042394472979], \
                                         [24.187203476683973, 2.4970022673408536, 2.7272769326695716, 5.14139220925684], \
                                         [3.1383721252068657, 3.7387409500864055, 5.4078774438507038, 2.6037751359723482], \
                                         [8.4478301876452306, 5.0320403406718492, 2.8834212079291985, 2.9415227772920427], \
                                         [3.1426495531487646, 4.0212439378901861, 0.60134852594003851, 1.7306126149977454], \
                                         [15.635374455133983, 3.2520409675251392, 0.48800515908613024, 6.194364208177519], \
                                         [79.02112152577908, 1.6997190910449294, 4.7100928164230398, 18.54561945000215], \
                                         [2.4147686613391968, 4.77983081183657, 5.6427304713913555, 3.6184518253194233], \
                                         [5.0053135736819581, 7.5224457127374018, 10.501213860336557, 14.976120894135667] ]
        else:
            # 200 training points
            initial_hyperparams = \
                                    [    [158.04660629989138, 1.2683731889725351, 66.115010215389162, 49.467620758110257, 2.1003587537731203, 148.19413243571267], \
                                         [65.998503029331715, 0.96182325466796015, 76.537506649529078, 8.7072874430795792, 2.7122005803599105, 31.811834175256053], \
                                         [7.2059309204166064, 248.71972897462479, 640.00239325817472, 168.88943619928102, 90.693076836996767, 262.04071654280534], \
                                         [6.0223358705627561, 5.7983854605769629, 7.151322371121573, 98.235863530785863, 108.97929055450719, 3.4049522081768151], \
                                         [2.8661028631425438, 202.71822125606951, 71.427958520531689, 67.841056622412466, 396.4966008975731, 8.7928316190417863], \
                                         [4.2274366302519937, 2.6787248415957619, 155.52642217518203, 12.902809348832989, 78.595731986539263, 91.408564485980548], \
                                         [1.9655515448696013, 1.0078459070165113, 47.264505099891736, 90.096628012215518, 39.449679781547545, 12.422258217851045], \
                                         [3.1222954891045123, 2.4207646445523401, 238.89962477023275, 110.82548792960249, 96.535214824647056, 13.310640119621617], \
                                         [4.9184407605671376, 2.2582428037913012, 220.5347539837725, 98.366730030395019, 51.082771850913375, 29.11755699767917], \
                                         [46.140309605544743, 3.4122114646182515, 66.780957403339869, 56.618087864345419, 3.4707008796589727, 8.5854627686410119], \
                                         [4.3654723196710217, 4.8059179997026096, 190.44003623250873, 122.8813398402322, 95.879994772087869, 3.3765033232957848], \
                                         [4.4788217098193668, 10.934823441369865, 123.33728936692876, 382.04572230065276, 227.51098327130356, 79.977071466413577], \
                                         [4.2294957554111283, 1.3726514772357896, 208.19938841666934, 174.7937853783985, 10.037269529873239, 127.83511529606474], \
                                         [27.075374606311115, 1.814069988416549, 7.7469247716502938, 170.97327917010006, 40.037475847074973, 56.043993677737269], \
                                         [83.400751642781657, 1.6747947434483126, 17.596540717504126, 112.41698370483411, 4.6137167168846629, 42.078275936624557], \
                                         [3.0901973040826474, 12.319455336644344, 70.719245134201159, 140.9221718578234, 224.77657336458043, 4.57995836705937], \
                                         [7.0310415618402358, 211.04475504456954, 202.37890056818438, 261.95210773212585, 156.94082106951919, 511.27885698486267] ]

            # 17 training points
            initial_hyperparams = \
                                    [    [195.98487106074634, 6.7545874303906324, 329.16664672945632, 31.504964572830737, 1.9724700463442919, 21.940669950105065], \
                                         [95.829408268109617, 1.7949030403189434, 35.784549991952147, 1.8075392976819478, 10.31753783934046, 27.645539145569355], \
                                         [2.4679367048563332, 275.24358705136979, 636.92711361596116, 168.68139572408469, 87.432840499884747, 258.97633592547965], \
                                         [2.3420253385025367, 22.62310001101493, 20.298926088403892, 212.56013373337302, 209.02779685614902, 15.565081046326867], \
                                         [2.9748867112646384, 200.28885691135244, 1.1710790148367796, 52.732688292649328, 350.37993600980144, 6.3245982727950123], \
                                         [4.3591136780785931, 16.744013045345845, 111.84371029836002, 2.6271052576644083, 102.14193905175958, 143.95163894287077], \
                                         [2.6009968687635978, 7.8504445946888541, 279.91377309872604, 571.13535376074287, 51.648328357923504, 57.43636247986246], \
                                         [2.4669769583966881, 31.401111683458542, 1270.673818203853, 337.26773045092648, 368.74473748301619, 75.987016939102659], \
                                         [3.192504460520428, 4.0107134770438648, 266.75067564837178, 100.4706686565055, 2.2156505927398489, 46.563261471015231], \
                                         [2.7471255292767269, 5.9252925255696667, 66.230363001377597, 29.189047727296355, 4.7635337732844629, 9.4122327461445128], \
                                         [3.7276325071593375, 14.779658802248324, 197.5161462630272, 262.24241780595895, 101.47737618311446, 2.008726905058027], \
                                         [56.268937406820683, 2.4305349403533811, 199.24414072357104, 2050.7935600821183, 254.31421742040177, 419.97964986565347], \
                                         [2.1187879592937864, 7.151426051051188, 673.33446347673043, 367.65295120249033, 18.135417989990003, 55.041101285874831], \
                                         [94.324177544164343, 9.5192423889915716, 2.4718983405006063, 126.99001654949174, 28.768268229786504, 130.52935833526666], \
                                         [2.1644252127344439, 3.7891115623661094, 25.035857452933165, 150.28115119268051, 6.1879295261351688, 63.966209357387271], \
                                         [6.6898343574716925, 1.8055110508489738, 44.732688365789286, 12.947694755847623, 71.881458838567767, 7.7704768954169774], \
                                         [2.4864586001908822, 214.54661462714421, 223.8076780527073, 248.81280629025022, 153.15645218324957, 515.19782558493819] ]

        batch_config = gp.batch_start(optimiser_config, initial_hyperparams)
        logging.info('Using Batch Start Configuration')

    else:
        batch_config = optimiser_config
    
    # Obtain the response function
    responsefunction = gp.classifier.responses.get(responsename)

    # Train the classifier!
    logging.info('Learning...')
    if batchlearn:
        previous_history = np.load(filename_start)
        learned_classifier = list(previous_history['learned_classifier'])
        white_params = previous_history['white_params']
        batch_config = \
            gp.classifier.batch_start(optimiser_config, learned_classifier)
        Fqw = white_fn(Fq, white_params)
    else:
        learned_classifier = gp.classifier.learn(Fw, y, 
            kerneldef, responsefunction, batch_config, 
            multimethod = multimethod, approxmethod = approxmethod, 
            train = train, ftol = 1e-10)

    # Print the learnt kernel with its hyperparameters
    print_function = gp.describer(kerneldef)
    gp.classifier.utils.print_learned_kernels(print_function, 
        learned_classifier, y_unique)

    # Print the matrix of learned classifier hyperparameters
    logging.info('Matrix of learned hyperparameters')
    gp.classifier.utils.print_hyperparam_matrix(learned_classifier)

    """Classifier Prediction"""

    yq_pred, yq_mie, yq_lde = sea.model.predictions(learned_classifier, Fqw,
        fusemethod = fusemethod)
    yq_esd = gp.classifier.equivalent_standard_deviation(yq_lde)
    miss_ratio = sea.model.miss_ratio(yq_pred, yq_truth)
    yq_lde_mean = yq_lde.mean()
    yq_mie_mean = yq_mie.mean()
    yq_pred_hist, _ = np.histogram(yq_pred, bins = np.arange(23), density  = True)
    logging.info('Miss Ratio: {0:.2f}%'.format(100 * miss_ratio))
    logging.info('Average Marginalised Linearised Model Differential Entropy: '\
        '{0:.2f}'.format(yq_lde_mean))
    logging.info('Average Marginalised Information Entropy: '\
        '{0:.2f}'.format(yq_mie_mean))

    """Visualise Query Prediction and Entropy"""
    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        c = yq_truth, vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap,
        **map_kwargs)
    sea.vis.describe_plot(title = 'Synthetic Ground Truth Map', 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)
    fig.tight_layout()

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        c = yq_pred, vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap,
        **map_kwargs)
    sea.vis.describe_plot(
        title = 'Prediction Map [Miss Ratio: {0:.2f}\%]'.format(100 * miss_ratio), 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Habitat Labels', cticks = y_unique, cticklabels = y_names,
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)
    fig.tight_layout()

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        c = yq_mie, cmap = cm.coolwarm, colorcenter = 'none', 
        **map_kwargs)
    sea.vis.describe_plot(title = 'Prediction Information Entropy', 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Information Entropy',
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)
    fig.tight_layout()

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        c = np.log(yq_mie), cmap = cm.coolwarm, colorcenter = 'none', 
        **map_kwargs)
    sea.vis.describe_plot(title = 'Log Prediction Information Entropy', 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Information Entropy',
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)
    fig.tight_layout()

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        c = yq_lde, cmap = cm.coolwarm, colorcenter = colorcenter_lde, 
        **map_kwargs)
    sea.vis.describe_plot(title = 'Linearised Model Differential Entropy', 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Differential Entropy',
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)
    fig.tight_layout()

    fig = plt.figure(figsize = (19.2, 10.8))
    sea.vis.scatter(
        Xq[:, 0], Xq[:, 1], 
        c = yq_esd, cmap = cm.coolwarm, colorcenter = colorcenter_analysis, 
        **map_kwargs)
    sea.vis.describe_plot(title = 'Equivalent Standard Deviation', 
        xlabel = 'x [Eastings (km)]', ylabel = 'y [Northings (km)]', 
        clabel = 'Standard Deviation',
        vis_range = vis_range, aspect_equal = True, 
        fontsize = FONTSIZE, fontname = FONTNAME, ticksize = TICKSIZE, 
        axis_scale = 1e3)
    fig.tight_layout()
    
    # fig = plt.figure(figsize = (8.0, 6.0))
    # plt.bar()

    """Save Results"""

    if SAVE_RESULTS:
        gp.classifier.utils.save_all_figures(full_directory, 
            axis_equal = True, extension = 'eps', rcparams = rcparams)
        shutil.copy2('./%s' % FILENAME , full_directory)
        np.savez('%sinitialmodel.npz' % full_directory, 
                learned_classifier = learned_classifier,
                t_seed = T_SEED, q_seed = Q_SEED,
                n_train = n_train, n_query = n_query,
                i_train = i_train, i_query = i_query,
                yq_pred = yq_pred, yq_mie = yq_mie, yq_lde = yq_lde,
                white_params = white_params)
    if MODEL_ONLY:
        plt.show()
        return

    """Informative Seafloor Exploration: Setup"""
    xq_now = np.array([[START_POINT1, START_POINT2]])

    # if MISSION_LENGTH > 0:
    #     if METHOD in ['LMDE', 'MCPIE', 'AMPIE']:
    #         acquisition_name = METHOD
    #     else:
    #         acquisition_name = 'AMPIE'

    #     xq_now = sea.explore.compute_new_starting_location(start_indices, Xq, Fqw, 
    #                     learned_classifier, acquisition = acquisition_name)

    xq_now = feature_fn.closest_locations(xq_now)
    horizon = HORIZON
    h_steps = H_STEPS

    if GREEDY or (METHOD == 'RANDOM') or (METHOD == 'FIXED'):
        horizon /= h_steps
        h_steps /= h_steps 

    if (METHOD == 'LMDE') or (METHOD == 'MCPIE'):
        theta_bound = np.deg2rad(20)
        theta_bounds = np.linspace(theta_bound, np.deg2rad(60), num = h_steps) 
        theta_stack_low  = -theta_bounds
        theta_stack_high = +theta_bounds
        xtol_rel = 1e-2
        ftol_rel = 1e-3
    else:
        theta_bound = np.deg2rad(270)
        theta_bounds = theta_bound * np.ones(h_steps)
        theta_stack_low  = -theta_bounds
        theta_stack_high = +theta_bounds
        xtol_rel = 1e-1
        ftol_rel = 1e-1
    ctol = 1e-10

    if METHOD == 'MCPIE':
        xtol_rel = 1e-1
        ftol_rel = 1e-1        

    theta_stack_init = np.deg2rad(0) * np.ones(h_steps)
    theta_stack_init[0] = np.deg2rad(180)
    theta_stack_low[0] = 0.0
    theta_stack_high[0] = 2 * np.pi
    r = horizon/h_steps
    choice_walltime = 1500.0

    k_step = 1
    m_step = 1

    bound = 100

    assert k_step == 1

    """Informative Seafloor Exploration: Initialisation"""
    # The observed data till now
    X_now = X.copy()
    y_now = y.copy()

    # Observe the current location
    i_observe = sea.feature.closest_indices(xq_now, Xq)
    yq_now = yq_truth[i_observe]

    # Add the observed data to the training set
    X_now = np.concatenate((X_now, xq_now[[-1]]), axis = 0)
    y_now = np.append(y_now, yq_now)

    # Add the new location to the array of travelled coordinates
    xq1_nows = xq_now[:, 0]
    xq2_nows = xq_now[:, 1]
    yq_nows = yq_now.copy()

    # Plot the current situation
    fig1 = plt.figure(figsize = (19.2, 10.8))
    fig2 = plt.figure(figsize = (19.2, 10.8))
    fig3 = plt.figure(figsize = (19.2, 10.8))
    fig4 = plt.figure(figsize = (19.2, 10.8))
    fig5 = plt.figure(figsize = (19.2, 10.8))

    # Start exploring
    i_trials = 0
    n_trials = N_TRIALS
    miss_ratio_array = np.nan * np.ones(n_trials)
    yq_mie_mean_array = np.nan * np.ones(n_trials)
    yq_lde_mean_array = np.nan * np.ones(n_trials)
    entropy_opt_array = np.nan * np.ones(n_trials)
    yq_esd_mean_array = np.nan * np.ones(n_trials)

    if METHOD == 'FIXED':
        turns = np.random.normal(loc = 0, scale = np.deg2rad(30), size = n_trials)

        # turns[[49, 99, 149]] = np.deg2rad(-90.0)

        # turns = np.linspace(-np.deg2rad(10), np.deg2rad(0), num = n_trials)

        # turns = np.deg2rad(30) * np.sin(np.linspace(0, 20*np.pi, num = n_trials))
        # turns = np.linspace(np.deg2rad(60), np.deg2rad(0), num = n_trials)

    while i_trials < n_trials:

        if MISSION_LENGTH > 0:
            if ((i_trials + 1) % MISSION_LENGTH == 0):

                if METHOD in ['LMDE', 'MCPIE', 'AMPIE']:
                    acquisition_name = METHOD

                    xq_now = sea.explore.compute_new_starting_location(start_indices, Xq, Fqw, 
                        learned_classifier, acquisition = acquisition_name)

                else:
                    
                    xq_now = Xq[np.random.choice(start_indices, size = 1, replace = False)]


        if METHOD == 'FIXED':
            theta_stack_init[0] += turns[i_trials]
            theta_stack_init[0] = np.mod(theta_stack_init[0], 2 * np.pi)

        # Propose a path
        if m_step <= k_step:
            if METHOD == 'RANDOM':
                xq_path, theta_stack_opt, entropy_opt = \
                    sea.explore.random_path(theta_stack_init, r, xq_now[-1], 
                        learned_classifier, feature_fn, white_params, 
                        bound = bound, 
                        chaos = CHAOS)
            elif METHOD == 'FIXED':
                xq_path, theta_stack_opt, entropy_opt = \
                    sea.explore.fixed_path(theta_stack_init, r, xq_now[-1], 
                        learned_classifier, feature_fn, white_params,
                        bound = bound, 
                        current_step = i_trials, 
                        turns = turns)
            else:
                xq_path, theta_stack_opt, entropy_opt = \
                    sea.explore.optimal_path(theta_stack_init, r, xq_now[-1],
                        learned_classifier, feature_fn, white_params,
                        objective = METHOD,
                        turn_limit = theta_bound,
                        bound = bound,
                        theta_stack_low = theta_stack_low,
                        theta_stack_high = theta_stack_high,
                        walltime = choice_walltime,
                        xtol_rel = xtol_rel,
                        ftol_rel = ftol_rel,
                        ctol = ctol,
                        globalopt = False,
                        n_draws = N_DRAWS,
                        depth_penalty = DEPTH_PENALTY)
            logging.info('Optimal Joint Entropy: %.5f' % entropy_opt)

            m_step = M_STEP
            logging.info('Taking %d steps' % m_step)
        else:
            m_step -= 1
            theta_stack_opt = theta_stack_init.copy()
            xq_path = sea.explore.forward_path_model(theta_stack_init, 
                r, xq_now[-1])
            logging.info('%d steps left' % m_step)

        # Path steps into the proposed path
        xq_now = xq_path[:k_step]

        # Initialise the next path angles
        theta_stack_init = sea.explore.shift_path(theta_stack_opt, 
            k_step = k_step, theta_bounds = theta_bounds)
        np.clip(theta_stack_init, 
            theta_stack_low + 1e-4, theta_stack_high - 1e-4, 
            out = theta_stack_init)

        # Observe the current location
        i_observe = sea.feature.closest_indices(xq_now, Xq)
        yq_now = yq_truth[i_observe]

        # Add the observed data to the training set
        X_now = np.concatenate((X_now, xq_now), axis = 0)
        y_now = np.append(y_now, yq_now)

        # Add the new location to the array of travelled coordinates
        xq1_nows = np.append(xq1_nows, xq_now[:, 0])
        xq2_nows = np.append(xq2_nows, xq_now[:, 1])
        yq_nows = np.append(yq_nows, yq_now)

        # Update that into the model
        Fw_now, white_params = feature_fn(X_now)
        logging.info('Learning Classifier...')
        batch_config = \
            gp.classifier.batch_start(optimiser_config, learned_classifier)
        try:
            learned_classifier = gp.classifier.learn(Fw_now, y_now, 
                kerneldef, responsefunction, batch_config, 
                multimethod = multimethod, approxmethod = approxmethod,
                train = True, ftol = 1e-6)
        except Exception as e:
            logging.warning('Training failed: {0}'.format(e))
            try:
                learned_classifier = gp.classifier.learn(Fw_now, y_now, 
                    kerneldef, responsefunction, batch_config, 
                    multimethod = multimethod, approxmethod = approxmethod,
                    train = False, ftol = 1e-6)
            except Exception as e:
                logging.warning('Learning also failed: {0}'.format(e))
                pass    
        logging.info('Finished Learning')

        # This is the finite horizon optimal route
        fqw_path = feature_fn(xq_path, white_params)
        xq1_path = xq_path[:, 0][k_step:]
        xq2_path = xq_path[:, 1][k_step:]
        yq_path = gp.classifier.classify(gp.classifier.predict(fqw_path, 
            learned_classifier), y_unique)[k_step:]

        """ Computing Analysis Maps """
        Fqw = white_fn(Fq, white_params)

        yq_pred, yq_mie, yq_lde = \
            sea.model.predictions(learned_classifier, Fqw, 
                fusemethod = fusemethod)
        yq_esd = gp.classifier.equivalent_standard_deviation(yq_lde)
        miss_ratio = sea.model.miss_ratio(yq_pred, yq_truth)
        yq_mie_mean = yq_mie.mean()
        yq_lde_mean = yq_lde.mean()
        yq_esd_mean = yq_esd.mean()
        logging.info('Miss Ratio: {0:.2f}%'.format(100 * miss_ratio))
        logging.info('Average Marginalised Linearised Model Differential Entropy: '\
            '{0:.2f}'.format(yq_lde_mean))
        logging.info('Average Marginalised Information Entropy: '\
            '{0:.2f}'.format(yq_mie_mean))


        """ Save history """
        miss_ratio_array[i_trials] = miss_ratio
        yq_mie_mean_array[i_trials] = yq_mie_mean
        yq_lde_mean_array[i_trials] = yq_lde_mean
        yq_esd_mean_array[i_trials] = yq_esd_mean
        entropy_opt_array[i_trials] = entropy_opt

        # Find the bounds of the entropy predictions
        vmin1 = yq_lde.min()
        vmax1 = yq_lde.max()
        vmin2 = yq_mie.min()
        vmax2 = yq_mie.max()
        vmin3 = yq_esd.min()
        vmax3 = yq_esd.max()

        logging.info('Plotting...')

        if ((i_trials + 1) % MISSION_LENGTH == 0) or ((i_trials + 2) % MISSION_LENGTH == 0) or (i_trials <= 10) or (((i_trials + 1) % SAVE_TRIALS) == 0):
            SAVE_EPS = True
        else:
            SAVE_EPS = False

        """ Linearised Entropy Map """

        # Prepare Figure 1
        plt.figure(fig1.number)
        plt.clf()
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            c = yq_lde, cmap = cm.coolwarm, colorcenter = colorcenter_lde,
            **map_kwargs)
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
        if MISSION_LENGTH == 0:
            sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
        else:
            sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linestyle = '--', linewidth = 1)
            xq1_nows_split = sea.vis.split_array(xq1_nows, MISSION_LENGTH)
            xq2_nows_split = sea.vis.split_array(xq2_nows, MISSION_LENGTH)
            [sea.vis.plot(xq1_nows_split[i], xq2_nows_split[i], c = 'k', linewidth = 2) for i in range(xq1_nows_split.shape[0])]
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
        if SAVE_EPS:
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
        if SAVE_EPS:
            plt.savefig('%slde_propose%d.eps' 
                % (full_directory, i_trials + 1))

        """ True Entropy Map """

        # Prepare Figure 3
        plt.figure(fig3.number)
        plt.clf()
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            c = yq_mie, cmap = cm.coolwarm, colorcenter = colorcenter_analysis,
            **map_kwargs)
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
        if MISSION_LENGTH == 0:
            sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
        else:
            sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linestyle = '--', linewidth = 1)
            xq1_nows_split = sea.vis.split_array(xq1_nows, MISSION_LENGTH)
            xq2_nows_split = sea.vis.split_array(xq2_nows, MISSION_LENGTH)
            [sea.vis.plot(xq1_nows_split[i], xq2_nows_split[i], c = 'k', linewidth = 2) for i in range(xq1_nows_split.shape[0])]
        sea.vis.scatter(xq_now[:, 0], xq_now[:, 1], c = yq_now, s = 120, 
            vmin = y_unique[0], vmax = y_unique[-1], 
            cmap = mycmap)

        # Save the plot
        fig3.tight_layout()
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.savefig('%smie%d.png' 
            % (full_directory, i_trials + 1))
        if SAVE_EPS:
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
        if SAVE_EPS:
            plt.savefig('%smie_propose%d.eps' 
                % (full_directory, i_trials + 1))

        """ Class Prediction Map """

        # Prepare Figure 4
        plt.figure(fig4.number)
        plt.clf()
        sea.vis.scatter(
            Xq[:, 0], Xq[:, 1], 
            c = yq_pred, vmin = y_unique[0], vmax = y_unique[-1], cmap = mycmap,
            **map_kwargs)
        sea.vis.describe_plot(
            title = 'Prediction Map [Miss Ratio: {0:.2f}\%]'.format(
                100 * miss_ratio), 
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
        if MISSION_LENGTH == 0:
            sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linewidth = 2)
        else:
            sea.vis.plot(xq1_nows, xq2_nows, c = 'k', linestyle = '--', linewidth = 1)
            xq1_nows_split = sea.vis.split_array(xq1_nows, MISSION_LENGTH)
            xq2_nows_split = sea.vis.split_array(xq2_nows, MISSION_LENGTH)
            [sea.vis.plot(xq1_nows_split[i], xq2_nows_split[i], c = 'k', linewidth = 2) for i in range(xq1_nows_split.shape[0])]
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
        if SAVE_EPS:
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
        if SAVE_EPS:
            plt.savefig('%spred_propose%d.eps' 
                % (full_directory, i_trials + 1))

        # Prepare Figure 5
        plt.figure(fig5.number)
        plt.clf()
        fontsize = 24
        ticksize = 14

        steps_array = np.arange(i_trials + 1) + 1
        ax = plt.subplot(4, 1, 1)
        plt.plot(steps_array, 100 * miss_ratio_array[:(i_trials + 1)])
        plt.title('Percentage of Prediction Misses', fontsize = fontsize)
        plt.ylabel('Misses (\%)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(4, 1, 2)
        plt.plot(steps_array, yq_lde_mean_array[:(i_trials + 1)])
        plt.title('Average Marginalised Differential Entropy', 
            fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(4, 1, 3)
        plt.plot(steps_array, yq_mie_mean_array[:(i_trials + 1)])
        plt.title('Average Marginalised Information Entropy', 
            fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)
        ax.set_xticklabels( () )

        ax = plt.subplot(4, 1, 4)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.plot(steps_array, entropy_opt_array[:(i_trials + 1)])
        plt.title('Entropy Metric of Proposed Path', fontsize = fontsize)
        plt.ylabel('Entropy (nats)', fontsize = fontsize)

        plt.xlabel('Steps', fontsize = fontsize)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize) 
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize) 

        # Save the plot
        fig5.tight_layout()
        plt.savefig('%shistory%d.png' 
            % (full_directory, i_trials + 1))
        logging.info('Plotted and Saved Iteration')
    
        # Move on to the next step
        i_trials += 1

        np.savez('%shistory%d.npz' % (full_directory, i_trials), 
            learned_classifier = learned_classifier,
            miss_ratio_array = miss_ratio_array,
            yq_lde_mean_array = yq_lde_mean_array,
            yq_mie_mean_array = yq_mie_mean_array,
            entropy_opt_array = entropy_opt_array,
            yq_esd_mean_array = yq_esd_mean_array,
            t_seed = T_SEED, q_seed = Q_SEED,
            n_train = n_train, n_query = n_query,
            i_train = i_train, i_query = i_query,
            yq_lde = yq_lde,
            yq_mie = yq_mie,
            yq_pred = yq_pred,
            white_params = white_params,
            X_now = X_now,
            Fw_now = Fw_now,
            y_now = y_now,
            xq1_path = xq1_path,
            xq2_path = xq2_path,
            fqw_path = fqw_path,
            yq_path = yq_path,
            xq1_nows = xq1_nows,
            xq2_nows = xq2_nows,
            yq_nows = yq_nows,
            vis_range = vis_range,
            colorcenter_analysis = colorcenter_analysis,
            colorcenter_lde = colorcenter_lde,
            y_unique = y_unique,
            mycmap = mycmap,
            i_trials = i_trials,
            theta_stack_opt = theta_stack_opt,
            theta_stack_init = theta_stack_init,
            xq_path = xq_path,
            xq_now = xq_now,
            yq_now = yq_now,
            i_observe = i_observe,
            horizon = horizon,
            h_steps = h_steps,
            r = r,
            y_names = y_names,
            FONTSIZE = FONTSIZE,
            FONTNAME = FONTNAME,
            TICKSIZE = TICKSIZE,
            SAVE_TRIALS = SAVE_TRIALS)

        logging.info('White Params: {0}'.format(white_params))

    np.savez('%shistory.npz' % full_directory, 
                learned_classifier = learned_classifier,
                miss_ratio_array = miss_ratio_array,
                yq_lde_mean_array = yq_lde_mean_array,
                yq_mie_mean_array = yq_mie_mean_array,
                entropy_opt_array = entropy_opt_array,
                yq_esd_mean_array = yq_esd_mean_array,
                t_seed = T_SEED, q_seed = Q_SEED,
                n_train = n_train, n_query = n_query,
                i_train = i_train, i_query = i_query,
                yq_lde = yq_lde,
                yq_mie = yq_mie,
                yq_pred = yq_pred,
                white_params = white_params,
                X_now = X_now,
                Fw_now = Fw_now,
                y_now = y_now,
                xq1_path = xq1_path,
                xq2_path = xq2_path,
                fqw_path = fqw_path,
                yq_path = yq_path,
                xq1_nows = xq1_nows,
                xq2_nows = xq2_nows,
                yq_nows = yq_nows,
                vis_range = vis_range,
                colorcenter_analysis = colorcenter_analysis,
                colorcenter_lde = colorcenter_lde,
                y_unique = y_unique,
                mycmap = mycmap,
                i_trials = i_trials,
                theta_stack_opt = theta_stack_opt,
                theta_stack_init = theta_stack_init,
                xq_path = xq_path,
                xq_now = xq_now,
                yq_now = yq_now,
                i_observe = i_observe,
                horizon = horizon,
                h_steps = h_steps,
                r = r,
                y_names = y_names,
                FONTSIZE = FONTSIZE,
                FONTNAME = FONTNAME,
                TICKSIZE = TICKSIZE,
                SAVE_TRIALS = SAVE_TRIALS)

    plt.show()

if __name__ == "__main__":
    main()
