"""
Test 'pathplanning'

Author: Kelvin
"""

import numpy as np
import computers.unsupervised.whitening as pre
import matplotlib.pyplot as plt
from matplotlib import cm
import computers.gp as gp
import pltutils

def fullprint(*args, **kwargs):
    from pprint import pprint
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)

def kerneldef(composer):
    h = composer.hyper
    k = composer.kernel

    # Sensitivity
    a = h(0.001, 1, 1)
    # b = h(0.001, 1, 0.001)
    # Length Scale
    l = h(0.01, 2, 0.1)
    # l1 = h(0.01, 2, 2)
    # l2 = h(0.01, 2, 2)

    return(a*k('gaussian', l)) #  + (1 - a)*k('matern3on2', [l1, l2]))

def main():
    """
    main
    """

    """
    Demonstration Parameters
    Note: Everything the user should set is here
    """

    # Feature Generation Parameters
    nTrain = 500
    nQuery = 500
    nDims  = 2   # <- This script visualises the 2D case (changing this will break visualisation)

    # If demonstrating multiclass classification, choose your method: 'OVA' or 'AVA'
    # This will be ignored if we're doing binary classification
    method = 'AVA' # 'AVA' or 'OVA'
    fusemethod = 'EXCLUSION' # 'HACK' or 'EXCLUSION'
    entropyThreshold = 0.9

    # Your test range
    yourTestRangeMin = -1.75
    yourTestRangeMax = +1.75

    # Your decision boundaries 
    yourDecisionBoundary1 = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * (0.9*(x1 + 1)**2 + x2**2/2) < 1.6) & ((x1 + x2) < 1.4)
    yourDecisionBoundary2 = lambda x1, x2: (((x1 - 1)**2 + x2**2/4) * (0.9*(x1 + 1)**2 + x2**2/2) > 0.2) & (x1**2 + x2**2 > 0.3)
    yourDecisionBoundary3 = lambda x1, x2: ((x1 + x2) < 1.8)
    yourDecisionBoundary4 = lambda x1, x2: ((x1 - 0.75)**2 + (x2 + 0.8)**2 > 0.3**2)
    yourDecisionBoundary  = [yourDecisionBoundary1, yourDecisionBoundary2, yourDecisionBoundary3, yourDecisionBoundary4]
    # yourDecisionBoundary = yourDecisionBoundary1

    # Your colormap for plotting labels
    mycmap = cm.jet

    """
    Demonstration Data Generation
    """

    #### Create training data
    np.random.seed(200)
    X = np.random.uniform(yourTestRangeMin, yourTestRangeMax, size = (nTrain, nDims))
    X = X[np.argsort(X[:, 0])] # <-Sorts the training points with respect to the first feature, in case we are plotting 1D data
    x1 = X[:, 0]
    x2 = X[:, 1]

    y = pltutils.makeDecision(X, yourDecisionBoundary)

    # Obtain unique labels for plotting later
    yUnique = np.unique(y)

    ### Create query (true) data
    Xq = np.random.uniform(yourTestRangeMin, yourTestRangeMax, size = (nQuery, nDims))
    Xq = Xq[np.argsort(Xq[:, 0])] # <-Sorts the query points with respect to the first feature, in case we are plotting 1D data

    # xqStart = np.array([1.5, 1.5])
    # Xq = np.concatenate((Xq, xqStart[:, np.newaxis]), axis = 0)
    # Xq = np.concatenate((Xq, np.array([[0.925, 0.485]])), axis = 0)
    xq1 = Xq[:, 0]
    xq2 = Xq[:, 1]

    yq = pltutils.makeDecision(Xq, yourDecisionBoundary)

    """
    Training Set Visualisation (before we continue to train)
    """

    fig = plt.figure()
    pltutils.visualiseDecisionBoundary(yourTestRangeMin, yourTestRangeMax, yourDecisionBoundary)
    
    plt.scatter(x1, x2, c = y, cmap = mycmap)
    plt.title('Training Labels (close this to start training)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(yUnique)
    cbar.set_ticklabels(yUnique)
    plt.xlim((yourTestRangeMin, yourTestRangeMax))
    plt.ylim((yourTestRangeMin, yourTestRangeMax))
    plt.show()

    """
    Kernel Composition
    """

    ### Compose Kernel
    composer = gp.kernel.Composer()
    mykernel = lambda : kerneldef(composer)

    # We can automatically extract the upper and lower theta vectors
    myKernelFn = composer.make(mykernel)  # compose a callable covariance function
    myPrintFn = lambda hyperparams: composer.describe(mykernel, hyperparams)
    
    """
    Classifier Training
    """

    # Set up optimisation
    learningHyperparams = gp.LearningParams()
    learningHyperparams.sigma = composer.range(mykernel)
    learningHyperparams.walltime = 60.0

    # Training
    print('===Begin Classifier Training===')
    (hyperparams, learnedMemories) = gp.classifier.learn(X, y, kerneldef, gp.logistic, learningHyperparams, train = True, ftol = 1e-10, verbose = True, method = method)

    """
    Classifier Training Results
    """
    finalKernelName = ''

    # Print the results
    if isinstance(learnedMemories, list):

        nResults = len(learnedMemories)

        for iResults in range(nResults):

            iClass1 = learnedMemories[iResults].cache.get('iClass')
            iClass2 = learnedMemories[iResults].cache.get('jClass')

            class1 = yUnique[iClass1]

            if iClass2 == -1:
                class2 = 'all'
                descript = '(Labels %d v.s. %s)' % (class1, class2)
            else:
                class2 = yUnique[iClass2]
                descript = '(Labels %d v.s. %d)' % (class1, class2)

            finalKernelName += 'Final Kernel %s: %s\n' % (descript, myPrintFn(hyperparams[iResults]))

    else:

        finalKernelName = 'Final Kernel: %s\n' % (myPrintFn(hyperparams))
    print(finalKernelName)

    """
    Classifier Prediction
    """

    # Prediction
    myPredictionFunction = lambda Xq: gp.classifier.predict(Xq, X, learnedMemories, kerneldef, fusemethod = fusemethod)
    yqProb = myPredictionFunction(Xq)
    yqPred = gp.distribution2modeclass(y, yqProb)
    yqEntropy = gp.entropyClassifier(yqProb)

    np.set_printoptions(threshold = np.inf)
    print('===Prediction Results===')
    print('---Prediction Features---')
    print(Xq)
    print('---Predicted Probabilities---')
    print(yqProb.T)
    print('---Predicted Labels---')
    print(yqPred)
    print('---Entropy---')
    print(yqEntropy)

    """
    Path Planning
    """

    XqPath = pathplanner(xqStart, myPredictionFunction)








    """
    THE GAP BETWEEN ANALYSIS AND PLOTS
    """







    """
    Classifier Prediction Results (Plots)
    """

    print('Plotting... please wait')

    """
    Plot: Training Set
    """

    # Training
    fig = plt.figure()
    pltutils.visualiseDecisionBoundary(yourTestRangeMin, yourTestRangeMax, yourDecisionBoundary)
    
    plt.scatter(x1, x2, c = y, cmap = mycmap)
    plt.title('Training Labels')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(yUnique)
    cbar.set_ticklabels(yUnique)
    plt.xlim((yourTestRangeMin, yourTestRangeMax))
    plt.ylim((yourTestRangeMin, yourTestRangeMax))

    """
    Plot: Prediction Entropy
    """

    # Query (Entropy)
    fig = plt.figure()
    pltutils.visualiseEntropy(yourTestRangeMin, yourTestRangeMax, myPredictionFunction, entropyThreshold = entropyThreshold)

    plt.title('Prediction Entropy')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()

    """
    Plot: Prediction Entropy onto Training Set
    """

    # Query (Entropy) and Training Set
    fig = plt.figure()
    pltutils.visualiseEntropy(yourTestRangeMin, yourTestRangeMax, myPredictionFunction, entropyThreshold = entropyThreshold)

    plt.title('Prediction Entropy and Training Set')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()

    plt.scatter(x1, x2, c = y, cmap = mycmap)
    plt.xlim((yourTestRangeMin, yourTestRangeMax))
    plt.ylim((yourTestRangeMin, yourTestRangeMax))

    """
    Plot: Prediction Labels
    """

    # Query (Prediction Map)
    fig = plt.figure()
    pltutils.visualisePrediction(yourTestRangeMin, yourTestRangeMax, lambda Xq: gp.distribution2modeclass(y, myPredictionFunction(Xq)), cmap = mycmap)

    plt.title('Prediction')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(yUnique)
    cbar.set_ticklabels(yUnique)

    """
    Plot: Prediction Probabilities for Multiclass Classifiers
    """

    # Visualise the final fused prediction probabilities (this produces multiple plots)
    if yUnique.shape[0] > 2:
        pltutils.visualisePredictionProbabilitiesMulticlass(yourTestRangeMin, yourTestRangeMax, myPredictionFunction, y)

    """
    Plot: Prediction Probabilities from Binary Classifiers
    """    
       
    # Query (Individual Maps)
    if isinstance(learnedMemories, list):

        for learnedResult in learnedMemories:

            myPredictionFunction = lambda Xq: gp.predictBinaryClassifier(Xq, learnedResult.X, learnedResult, kerneldef)

            fig = plt.figure()
            pltutils.visualisePredictionProbabilities(yourTestRangeMin, yourTestRangeMax, myPredictionFunction)

            iClass1 = learnedResult.cache.get('iClass')
            iClass2 = learnedResult.cache.get('jClass')

            class1 = yUnique[iClass1]

            if iClass2 == -1:
                class2 = 'all'
                title = 'Prediction Probabilities from Binary Classifier (Labels %d v.s. %s)' % (class1, class2)
            else:
                class2 = yUnique[iClass2]
                title = 'Prediction Probabilities from Binary Classifier (Labels %d v.s. %d)' % (class1, class2)
            plt.title(title)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()

    """
    Plot: Query Predictions 
    """   

    # Query (Predicted)
    fig = plt.figure()
    pltutils.visualiseDecisionBoundary(yourTestRangeMin, yourTestRangeMax, yourDecisionBoundary)
    
    plt.scatter(xq1, xq2, c = yqPred, cmap = mycmap)
    plt.title('Predicted Query Labels')
    plt.xlabel('x1')
    plt.ylabel('x2')
    cbar = plt.colorbar()
    cbar.set_ticks(yUnique)
    cbar.set_ticklabels(yUnique)
    plt.xlim((yourTestRangeMin, yourTestRangeMax))
    plt.ylim((yourTestRangeMin, yourTestRangeMax))

    """
    Save Results and Figures
    """

    from time import gmtime, strftime
    import os
    import sys

    # Directory names
    figureDirectoryName = 'Figures/'
    thisTimeDirectoryName = '%s_%s_%s/' % (strftime("%Y%m%d_%H%M%S", gmtime()), method, fusemethod)
    fullDirectoryName = '%s%s' % (figureDirectoryName, thisTimeDirectoryName)
    
    # Create directories
    if not os.path.isdir(figureDirectoryName):
        os.mkdir(figureDirectoryName)
    if not os.path.isdir(fullDirectoryName):
        os.mkdir(fullDirectoryName)

    # Go through each figure and save them
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig('%sFigure%d.png' % (fullDirectoryName, i))

    print('Figures Saved')

    textfilename = '%slog.txt' % (fullDirectoryName)
    textfile = open(textfilename, 'w')

    sys.stdout = textfile
    print(finalKernelName)
    np.set_printoptions(threshold = np.inf)
    print('===Prediction Results===')
    print('---Prediction Features---')
    print(Xq)
    print('---Predicted Probabilities---')
    print(yqProb.T)
    print('---Predicted Labels---')
    print(yqPred)
    print('---Entropy---')
    print(yqEntropy)

    # Show everything!
    plt.show()

if __name__ == "__main__":
    main()
