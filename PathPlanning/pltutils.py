"""
module 'pltutils'
This module contains some convenient functions for plot utility

Author: Kelvin
"""

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import computers.gp as gp

"""
2D Mesh Utilities
"""

def mesh2data(Xmesh, Ymesh, Zmesh = None):
	""" 
	Converts data in meshgrid format to design matrix format
	'Zmesh' is an optimal argument that can be passed if you want
	to transform it into design matrix format by the same operation
	as those defined by 'Xmesh' and 'Ymesh'
	"""

	# Convert the meshgrids into design matrices
	(xLen, yLen) = Xmesh.shape
	n = xLen * yLen
	data = np.zeros((n, 2))
	data[:, 0] = np.reshape(Xmesh, n)
	data[:, 1] = np.reshape(Ymesh, n)

	# Return the design matrix or compute the design matrix for the last meshgrid if needed
	if Zmesh == None:
		return(data)
	else:
		z = np.reshape(Zmesh, n)
		return(data, z)

def data2mesh(data, meshshape, zs = None):
	""" 
	Converts data in design matrix format to meshgrid format
	'zs' is an optimal argument that can be passed if you want
	to transform it into meshgrid format by the same operation
	Note that 'zs' can be a list of data that 
	"""

	# The data needs to be just 2D for a meshgrid to make sense
	# Yes, ND meshgrids exists but this is 'pltutils' - we can only plot things with 2D mesh! (I think?)
	x = data[:, 0]
	y = data[:, 1]

	# Find the meshgrid representation of our 2D data
	Xmesh = np.reshape(x, meshshape)
	Ymesh = np.reshape(y, meshshape)

	# Return the meshgrids or compute more meshgrids if requested
	if zs == None:
		return(Xmesh, Ymesh)
	else:
		
		# Case 1: zs is just a numpy array
		try:

			# find the meshgrid representation of the numpy array if it is a 1D numpy array
			shape = zs.shape
			if len(shape) == 1:
				Zmesh = np.reshape(zs, meshshape)

			# Let this code break for now if this is ND numpy array with N > 1
			else:
				raise ValueError('zs needs to be 1D numpy arrays!')

		# Case 2: zs is a list of numpy arrays
		except Exception:

			# Go through the list and find the meshgrid representation of each numpy array
			Zmesh = zs.copy()
			for i in range(len(zs)):
				Zmesh[i] = np.reshape(zs[i], meshshape)

		return(Xmesh, Ymesh, Zmesh)

"""
2D Classification Visualisation Utilities
"""

def makeDecision(X, condition):
    """
    With 'condition(x1, x2)' as your decision boundary criteria (inequalities), find the label 'y' given features 'X'

    In the binary case, 'y' takes on values of -1 and 1.
    In the multiclass case, 'y' takes on values of 0, 1, 2, ..., M - 1 where M is the number of classes.
    In the multiclass case, condition' is a list of ineqaulities, and the decision boundary for the first label (0) is 
    obtained by the union of all the inqualities, where as the rest of the labels (1 to M - 1) is determined by the negation
    of each inquality.

    Arguments:
    	X        : Features in data design matrix format
    	condition: Function or list of functions that maps (x1, x2) to true or false based on an inquality
    Returns:
    	y        : Labels
    """

    # Extract the components - this only works for 2D case
    x1 = X[:, 0]
    x2 = X[:, 1]

    # If it is just one condition, create -1 and 1 labels for binary classification
    if callable(condition):

        # Create the labels of -1 and 1
        y = -np.ones(x1.shape)
        y[condition(x1, x2)] = 1

        # Return the labels
        return(y)

    # If it is multiple conditions, create labels 0, 1, 2, ..., M - 1
    elif isinstance(condition, list):

    	# Start with zero
        y = np.zeros(x1.shape)

        # For each extra condition, insert another label
        for i in range(len(condition)):

        	# Insert labels at places that do not satisfy the conditions
            y[~condition[i](x1, x2)] = i + 1

        # Return the labels
        return(y)

def visualiseDecisionBoundary(yourTestRangeMin, yourTestRangeMax, condition):
    """
    Visualises the decision boundary defined by 'condition(x1, x2)'

    In the multiclass case, condition' is a list of ineqaulities, and the decision boundary for the first label (0) is 
    obtained by the union of all the inqualities, where as the rest of the labels (1 to M - 1) is determined by the negation
    of each inquality.

    For now, the visualisation is done on a square area

    Arguments:
    	yourTestRangeMin: Minimum x and y boundary 
    	yourTestRangeMax: Maximum x and y boundary 
    	condition       : Function or list of functions that maps (x1, x2) to true or false based on an inquality   
    Returns:
    	contour	 		: The plot countour object
    """

    # Make sure our contours are solid black lines
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    # Create the query points for plotting
    xq = np.linspace(yourTestRangeMin, yourTestRangeMax, num = 200)
    (xq1Mesh, xq2Mesh) = np.meshgrid(xq, xq)
    Xq = mesh2data(xq1Mesh, xq2Mesh)
    yq = makeDecision(Xq, condition)

    # Convert that to a mesh
    yqMesh = data2mesh(Xq, xq1Mesh.shape, zs = yq)[-1]
    
    # Determine the levels of contours we will draw
    yqUnique = np.unique(yq)
    yqMin = yqUnique[0]
    yqMax = yqUnique[-1]
    nLevels = yqUnique.shape[0]
    levels = np.linspace(yqMin, yqMax, num = nLevels)
    
    contour = plt.contour(xq1Mesh, xq2Mesh, yqMesh, levels = levels, rstride = 1, cstride = 1, colors = 'k', linewidth = 0, antialiased = False)

    return(contour)

def visualisePredictionProbabilities(yourTestRangeMin, yourTestRangeMax, predictionFunction):
    """
    Visualises the predicted probabilities with the prediction function 'predictionFunction(Xq)'

    'predictionFunction' is usually obtained by using a lambda expression on the predictor function,
    in order to abstract away all the other arguments

    For now, the visualisation is done on a square area

    The probabilities can only be visualised for the binary case, so for the multiclass version,
    you would have multiple plots for each combination of binary classification problems in your setup

    Arguments:
    	yourTestRangeMin  : Minimum x and y boundary 
    	yourTestRangeMax  : Maximum x and y boundary 
    	predictionFunction: Prediction function that maps input query features 'Xq' to expected probabilities 'yqProb' (piq)
    Returns:
    	contour	 		  : The plot countour object
    	image             : The plot pcolormesh object
    """

    # Create the query points for plotting
    xq = np.linspace(yourTestRangeMin, yourTestRangeMax, num = 100)
    (xq1Mesh, xq2Mesh) = np.meshgrid(xq, xq)
    Xq = mesh2data(xq1Mesh, xq2Mesh)
    yqProb = predictionFunction(Xq)

    # Convert that to a mesh
    yqMesh = data2mesh(Xq, xq1Mesh.shape, zs = yqProb)[-1]
    
    # Visualise it
    levels = np.linspace(0, 1, num = 200)
    contour = plt.contour(xq1Mesh, xq2Mesh, yqMesh, np.linspace(0, 1, num = 3), rstride = 1, cstride = 1, cmap = cm.gray, linewidth = 0, antialiased = False)
    image = plt.pcolormesh(xq1Mesh, xq2Mesh, yqMesh, cmap = cm.coolwarm)
    return(contour, image)

def visualisePredictionProbabilitiesMulticlass(yourTestRangeMin, yourTestRangeMax, predictionFunction, yqReference):
    """
    Visualises the predicted probabilities with the prediction function 'predictionFunction(Xq)'

    'predictionFunction' is usually obtained by using a lambda expression on the predictor function,
    in order to abstract away all the other arguments

    For now, the visualisation is done on a square area

    This visualised the final fused predicted probabilities for each class, and so will produced 'nClass' figures, 
    where 'nClass' is the number of classes we have

    Arguments:
    	yourTestRangeMin  : Minimum x and y boundary 
    	yourTestRangeMax  : Maximum x and y boundary 
    	predictionFunction: Prediction function that maps input query features 'Xq' to expected probabilities 'yqProb' (piq)
    Returns:
    	contour	 		  : The plot countour object
    	image             : The plot pcolormesh object
    """

    # Create the query points for plotting
    xq = np.linspace(yourTestRangeMin, yourTestRangeMax, num = 100)
    (xq1Mesh, xq2Mesh) = np.meshgrid(xq, xq)
    Xq = mesh2data(xq1Mesh, xq2Mesh)
    yqProbs = predictionFunction(Xq)

    contours = []
    images = []

    yqUnique = np.unique(yqReference)

    i = 0
    for yqProb in yqProbs:

        plt.figure()
        # Convert that to a mesh
        yqMesh = data2mesh(Xq, xq1Mesh.shape, zs = yqProb)[-1]
        
        # Visualise it
        levels = np.linspace(0, 1, num = 200)
        contour = plt.contour(xq1Mesh, xq2Mesh, yqMesh, np.linspace(0, 1, num = 3), rstride = 1, cstride = 1, cmap = cm.gray, linewidth = 0, antialiased = False)
        image = plt.pcolormesh(xq1Mesh, xq2Mesh, yqMesh, cmap = cm.coolwarm)
        contours.append(contour)
        images.append(image)

        plt.title('Final Fused Prediction Probabilities of (Label %g)' % yqUnique[i])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
        i += 1

    return(contours, images)

def visualiseEntropy(yourTestRangeMin, yourTestRangeMax, predictionFunction, entropyThreshold = 0.9):
    """
    Visualises the entropy with the prediction function 'predictionFunction(Xq)'

    'predictionFunction' is usually obtained by using a lambda expression on the predictor function,
    in order to abstract away all the other arguments

    For now, the visualisation is done on a square area

    The probabilities can only be visualised for the binary case, so for the multiclass version,
    you would have multiple plots for each combination of binary classification problems in your setup

    Arguments:
    	yourTestRangeMin  : Minimum x and y boundary 
    	yourTestRangeMax  : Maximum x and y boundary 
    	predictionFunction: Prediction function that maps input query features 'Xq' to expected probabilities 'yqProb' (piq)
    Returns:
    	contour	 		  : The plot countour object
    	image             : The plot pcolormesh object
    """

    # Create the query points for plotting
    xq = np.linspace(yourTestRangeMin, yourTestRangeMax, num = 80)
    (xq1Mesh, xq2Mesh) = np.meshgrid(xq, xq)
    Xq = mesh2data(xq1Mesh, xq2Mesh)
    yqProb = predictionFunction(Xq)   
    yqEntropy = gp.classifier.entropy(yqProb)

    # Convert that to a mesh
    yqEntropyMesh = data2mesh(Xq, xq1Mesh.shape, zs = yqEntropy)[-1]
    
    # Visualise it
    # yqEntropyMin = yqEntropy.min()
    # yqEntropyMax = yqEntropy.max()
    # yqEntropyThreshold = entropyThreshold * (yqEntropyMax - yqEntropyMin) + yqEntropyMin

    yqEntropyThreshold = np.percentile(yqEntropy, 100 * entropyThreshold)

    contour = plt.contour(xq1Mesh, xq2Mesh, yqEntropyMesh, levels = [yqEntropyThreshold], rstride = 1, cstride = 1, cmap = cm.gray, linewidth = 0, antialiased = False)
    image = plt.pcolormesh(xq1Mesh, xq2Mesh, yqEntropyMesh, cmap = cm.coolwarm)
    return(contour, image)

def visualisePrediction(yourTestRangeMin, yourTestRangeMax, predictionClassFunction, cmap = cm.jet):
    """
    Visualises the predictions with the prediction function 'predictionClassFunction(Xq)'

    'predictionFunction' is usually obtained by using a lambda expression on the predictor function,
    in order to abstract away all the other arguments

    For now, the visualisation is done on a square area

    The probabilities can only be visualised for the binary case, so for the multiclass version,
    you would have multiple plots for each combination of binary classification problems in your setup

    Arguments:
    	yourTestRangeMin  : Minimum x and y boundary 
    	yourTestRangeMax  : Maximum x and y boundary 
    	predictionFunction: Prediction function that maps input query features 'Xq' to expected probabilities 'yqProb' (piq)
    Keyword Arguments:
    	cmap			  : The color map to be used for the image
    Returns:
    	contour	 		  : The plot countour object
    	image             : The plot pcolormesh object
    """

    # Make sure our contours are solid black lines
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    # Create the query points for plotting
    xq = np.linspace(yourTestRangeMin, yourTestRangeMax, num = 80)
    (xq1Mesh, xq2Mesh) = np.meshgrid(xq, xq)
    Xq = mesh2data(xq1Mesh, xq2Mesh)
    yqPred = predictionClassFunction(Xq)

    # Convert that to a mesh
    yqMesh = data2mesh(Xq, xq1Mesh.shape, zs = yqPred)[-1]

    # Determine the levels of contours we will draw
    yqUnique = np.unique(yqPred)
    yqMin = yqUnique[0]
    yqMax = yqUnique[-1]
    nLevels = yqUnique.shape[0]
    levels = np.linspace(yqMin, yqMax, num = nLevels)

    # Visualise it!
    contour = plt.contour(xq1Mesh, xq2Mesh, yqMesh, levels = levels, rstride = 1, cstride = 1, colors = 'k', linewidth = 0, antialiased = False)
    image = plt.pcolormesh(xq1Mesh, xq2Mesh, yqMesh, cmap = cmap)
    return(contour, image)