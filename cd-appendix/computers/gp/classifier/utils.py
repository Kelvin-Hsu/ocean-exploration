"""
module 'pltutils'
This module contains some convenient functions for plot utility

Author: Kelvin
"""

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import time
import logging

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2D Mesh Utilities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def mesh2data(Xmesh, Ymesh, Zmesh = None):
	""" 
	Converts data in meshgrid format to design matrix format
	'Zmesh' is an optimal argument that can be passed if you want
	to transform it into design matrix format by the same operation
	as those defined by 'Xmesh' and 'Ymesh'
	"""

	# Convert the meshgrids into design matrices
	(x_len, y_len) = Xmesh.shape
	n = x_len * y_len
	data = np.zeros((n, 2))
	data[:, 0] = np.reshape(Xmesh, n)
	data[:, 1] = np.reshape(Ymesh, n)

	# Return the design matrix or compute the design matrix for the last 
    # meshgrid if needed
	if Zmesh == None:
		return data 
	else:
		z = np.reshape(Zmesh, n)
		return data, z 

def data2mesh(data, meshshape, zs = None):
	""" 
	Converts data in design matrix format to meshgrid format
	'zs' is an optimal argument that can be passed if you want
	to transform it into meshgrid format by the same operation
	Note that 'zs' can be a list of data that 
	"""

	# The data needs to be just 2D for a meshgrid to make sense
	# Yes, ND meshgrids exists but this is 'pltutils' - we can only plot 
    # things with 2D mesh! (I think?)
	x = data[:, 0]
	y = data[:, 1]

	# Find the meshgrid representation of our 2D data
	Xmesh = np.reshape(x, meshshape)
	Ymesh = np.reshape(y, meshshape)

	# Return the meshgrids or compute more meshgrids if requested
	if zs == None:
		return Xmesh, Ymesh 
	else:
		
		# Case 1: zs is just a numpy array
		try:

			# find the meshgrid representation of the numpy array if it is a 
            # 1D numpy array
			shape = zs.shape
			if len(shape) == 1:
				Zmesh = np.reshape(zs, meshshape)

			# Let this code break for now if this is ND numpy array with N > 1
			else:
				raise ValueError('zs needs to be 1D numpy arrays!')

		# Case 2: zs is a list of numpy arrays
		except Exception:

			# Go through the list and find the meshgrid representation of 
            # each numpy array
			Zmesh = zs.copy()
			for i in range(len(zs)):
				Zmesh[i] = np.reshape(zs[i], meshshape)

		return Xmesh, Ymesh, Zmesh 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2D Classification Visualisation Utilities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def make_decision(X, condition):
    """
    With 'condition(x1, x2)' as your decision boundary 
    criteria (inequalities), find the label 'y' given features 'X'

    In the binary case, 'y' takes on values of -1 and 1.
    In the multiclass case, 'y' takes on values of 0, 1, 2, ..., M - 1 
    where M is the number of classes.
    In the multiclass case, 'condition' is a list of ineqaulities, 
    and the decision boundary for the first label (0) is 
    obtained by the intersection of all the inqualities, 
    where as the rest of the labels (1 to M - 1) is determined by the negation
    of each inquality.

    Arguments:
    	X        : Features in data design matrix format
    	condition: Function or list of functions that maps (x1, x2) to true 
                   or false based on an inquality
    Returns:
    	y        : Labels
    """

    if X.shape[1] != 2:
        raise ValueError('Data "X" needs to be 2D')

    # Extract the components - this only works for 2D case
    x1 = X[:, 0]
    x2 = X[:, 1]

    # If it is just one condition, create -1 and 1 labels for binary 
    # classification
    if callable(condition):

        # Create the labels of -1 and 1
        y = -np.ones(x1.shape)
        y[condition(x1, x2)] = 1

        # Return the labels
        return y.astype(int)

    # If it is multiple conditions, create labels 0, 1, 2, ..., M - 1
    elif isinstance(condition, list):

    	# Start with zero
        y = np.zeros(x1.shape)

        # For each extra condition, insert another label
        for i in range(len(condition)):

        	# Insert labels at places that do satisfy the conditions
            y[~condition[i](x1, x2)] = i + 1

        # Return the labels
        return y.astype(int)

    else:

        raise ValueError('"condition" needs to be a callable function or a \
            list of callable functions')

def generate_elliptical_decision_boundaries(ranges, 
    min_size = 0.1, max_size = 0.5, n_class = 2, n_ellipse = 30, n_dims = 2):
    """
    Randomly generates 2D decision boundaries for arbitary number of classes

    Arguments:
        ranges      : A tuple containing the minimum and maximum range

    Keyword Arguments:
        min_size    : Minimum size of the ellipses
        max_size    : Maximum size of the ellipses
        n_class     : Number of classes to be generated
        n_ellipse   : Number of ellipses to be generated for each class
        n_dims      : Number of dimensions (for now this must be 2)

    Returns:
        db          : A list of decision boundaries
    """
    ellipse = lambda x1, x2, A: (((x1-A[0])/A[2])**2+((x2-A[1])/A[3])**2<1)

    n_db = n_class - 1
    size = (n_db, n_ellipse, n_dims)

    P = np.random.uniform(ranges[0], ranges[1], size = size)
    B = np.random.uniform(min_size, max_size, size = size)
    A = np.concatenate((P, B), axis = 2)

    db_constructor = lambda i: lambda x1, x2: np.array([ellipse(x1, x2, a) 
        for a in A[i]]).sum(axis = 0) > 0 
    return [db_constructor(i) for i in range(n_db)]

def visualise_decision_boundary(ax, range_min, range_max, condition, 
    n_points = 1000):
    """
    Visualises the decision boundary defined by 'condition(x1, x2)'

    In the multiclass case, condition' is a list of ineqaulities, and the 
    decision boundary for the first label (0) is 
    obtained by the union of all the inqualities, where as the rest of the 
    labels (1 to M - 1) is determined by the negation
    of each inquality.

    For now, the visualisation is done on a square area

    Arguments:
    	range_min: Minimum x and y boundary 
    	range_max: Maximum x and y boundary 
    	condition: Function or list of functions that maps (x1, x2) 
                             to true or false based on an inquality   
    Returns:
    	contour  : The plot countour object
    """

    # Make sure our contours are solid black lines
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    # Create the query points for plotting
    xq = np.linspace(range_min, range_max, num = n_points)
    (xq1_mesh, xq2_mesh) = np.meshgrid(xq, xq)
    Xq = mesh2data(xq1_mesh, xq2_mesh)
    yq = make_decision(Xq, condition)


    # Convert that to a mesh
    yq_mesh = data2mesh(Xq, xq1_mesh.shape, zs = yq)[-1]
    
    # Determine the levels of contours we will draw
    yq_unique = np.unique(yq)
    yq_min = yq_unique[0]
    yq_max = yq_unique[-1]
    n_levels = yq_unique.shape[0]
    levels = np.linspace(yq_min, yq_max, num = n_levels)
    
    plt.sca(ax)
    contour = plt.contour(xq1_mesh, xq2_mesh, yq_mesh, 
        levels = levels, rstride = 1, cstride = 1, colors = 'k', 
        linewidth = 0, antialiased = False)

    return yq, Xq, contour 

def query_map(ranges, n_points = 100):
    """
    Creates query points in data format corresponding to a 2D mesh

    Arguments:
        ranges      : A tuple containing the minimum and maximum range

    Keyword Arguments:
        n_points    : Number of grid points in each dimension

    Returns:
        Xq          : The query points in data format
    """
    assert isinstance(ranges, tuple)

    if len(ranges) == 2:
        range_min_1, range_max_1 = ranges
        range_min_2, range_max_2 = ranges
    elif len(ranges) == 4:
        range_min_1, range_max_1, range_min_2, range_max_2 = ranges

    xq1 = np.linspace(range_min_1, range_max_1, num = n_points)
    xq2 = np.linspace(range_min_2, range_max_2, num = n_points)
    (xq1_mesh, xq2_mesh) = np.meshgrid(xq1, xq2)
    Xq = mesh2data(xq1_mesh, xq2_mesh)
    return Xq

def visualise_map(ax, yq, ranges, n_points = None,
    boundaries = False, threshold = None, levels = None, **kwargs):
    """
    Visualises any query maps generated by 'query_map'

    Arguments:
        yq          : The values to be visualised
        ranges      : A tuple containing the minimum and maximum range

    Keyword Arguments:
        n_points    : Number of grid points in each dimension
        boundaries  : Put boundary contours between distinct/discrete values
        threshold   : Put threshold contours at this threshold quantile
        levels      : Directly specify the contour levels
        **kwargs    : All other keyword arguments for a standard scatter plot
    Returns:
        image       : The matplotlib plot object for the image
        contour     : The matplotlib plot object for the contour, if any
    """
    assert isinstance(ranges, tuple)

    if len(ranges) == 2:
        range_min_1, range_max_1 = ranges
        range_min_2, range_max_2 = ranges
    elif len(ranges) == 4:
        range_min_1, range_max_1, range_min_2, range_max_2 = ranges

    if n_points == None:
        n_points = int(np.sqrt(yq.shape[0]))
    xq1 = np.linspace(range_min_1, range_max_1, num = n_points)
    xq2 = np.linspace(range_min_2, range_max_2, num = n_points)
    (xq1_mesh, xq2_mesh) = np.meshgrid(xq1, xq2)
    Xq = mesh2data(xq1_mesh, xq2_mesh)

    yq_mesh = data2mesh(Xq, xq1_mesh.shape, zs = yq)[-1]

    contour = None

    if boundaries:
        yq_unique = np.unique(yq)
        yq_min = yq_unique[0]
        yq_max = yq_unique[-1]
        n_levels = yq_unique.shape[0]
        levels = np.linspace(yq_min, yq_max, num = n_levels)

    if threshold is not None:
        y_threshold = np.percentile(yq, 100 * levels)
        levels = [y_threshold]

    plt.sca(ax)
    if levels is not None:
        contour = plt.contour(xq1_mesh, xq2_mesh, yq_mesh, 
            levels = levels, rstride = 1, cstride = 1, 
            colors = 'k', linewidth = 0, antialiased = False)

    image = plt.pcolormesh(xq1_mesh, xq2_mesh, yq_mesh, **kwargs)
    
    return image, contour 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Printer Utilities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def binary_classifier_name(learned_classifier, y_unique):

    y_unique = np.unique(y_unique)
    i_class = learned_classifier.cache.get('i_class')
    j_class = learned_classifier.cache.get('j_class')
    class1 = y_unique[i_class]
    if j_class == -1:
        class2 = 'all'
        descript = '(Labels %d v.s. %s)' % (class1, class2)
    else:
        class2 = y_unique[j_class]
        descript = '(Labels %d v.s. %d)' % (class1, class2)
    return descript

def print_learned_kernels(print_function, learned_classifier, y_unique):
    """
    Prints learned classifier kernels in detail

    Arguments:
        print_function      : Printer function obtained through
                                print_function = gp.describer(kerneldef)
        learned_classifier  : An instance of Classifier Memory or a list thereof
        y_unique            : An np.array of unique labels
    Returns:
        None
    """
    y_unique = np.unique(y_unique)
    kernel_descriptions = '\n'
    if isinstance(learned_classifier, list):
        n_results = len(learned_classifier)
        for i in range(n_results):
            descript = binary_classifier_name(learned_classifier[i], y_unique)
            kernel_descriptions += 'Final Kernel %s: %s \t | \t '\
                'Log Marginal Likelihood: %.8f \n' \
                % (descript, print_function(
                    [learned_classifier[i].hyperparams]), 
                     learned_classifier[i].logmarglik)
    else:
        kernel_descriptions = 'Final Kernel: %s\n' \
            % (print_function([learned_classifier.hyperparams]))
    logging.info(kernel_descriptions)

def print_hyperparam_matrix(learned_classifier, precision = 0):

    # TO DO: Allow precision control
    
    if isinstance(learned_classifier, list):
        n_classifiers = len(learned_classifier)
        matrixstring = ''
        for i_classifiers in range(n_classifiers):
            if i_classifiers == 0:
                matrixstring += '\n[\t {0}, \\\n'.format(
                    learned_classifier[i_classifiers].hyperparams)
            elif i_classifiers == n_classifiers - 1:
                matrixstring += ' \t {0} ]\n'.format(
                    learned_classifier[i_classifiers].hyperparams)
            else:
                matrixstring += ' \t {0}, \\\n'.format(
                    learned_classifier[i_classifiers].hyperparams)
        logging.info(matrixstring)
    else:
        logging.info(learned_classifier.hyperparams)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Miscellaneous
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def create_directories(save_directory, 
    home_directory = 'Figures/', append_time = True, casual_format = False):

    # Directory names
    if append_time:
        if casual_format:
            time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", time.gmtime())
        else:
            time_string = time.strftime("%Y%m%d_%H%M%S__", time.gmtime())
        save_directory = time_string + save_directory
    full_directory = '%s%s' \
        % (home_directory, save_directory)

    # Create directories
    if not os.path.isdir(home_directory):
        os.mkdir(home_directory)
    if not os.path.isdir(full_directory):
        os.mkdir(full_directory)
    return full_directory

def save_all_figures(full_directory, 
    axis_equal = True, tight = True, extension = 'eps', rcparams = None):

    if rcparams is not None:
        plt.rc_context(rcparams)

    # Go through each figure and save them
    for i in plt.get_fignums():
        fig = plt.figure(i)
        if axis_equal:
            plt.gca().set_aspect('equal', adjustable = 'box')
        if tight:
            fig.tight_layout()
        fig.savefig('%sFigure%d.%s' % (full_directory, i, extension))
    logging.info('Figures Saved.')

def plot_circle(center, radius, num = 100, **kwargs):

    t = np.linspace(0, 2*np.pi, num = num)
    x = radius * np.cos(t) + center[0]
    y = radius * np.sin(t) + center[1]
    return plt.plot(x, y, **kwargs)
