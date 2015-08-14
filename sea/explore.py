"""
Informative Seafloor Exploration
"""
import numpy as np
import nlopt
from computers import gp
import logging
import time
from scipy.spatial.distance import cdist

def optimal_path(theta_stack_init, r, x, memory, feature_fn, white_params,
    objective = 'LDE', turn_limit = np.deg2rad(30), bound = 100,
    theta_stack_low = None, theta_stack_high = None,
    walltime = None, xtol_rel = 0, ftol_rel = 0, ctol = 1e-6, 
    globalopt = False, n_draws = 5000):

    # Select the approach objective
    if objective == 'LDE':

        def objective(theta_stack, grad):
            return path_linearised_entropy_model(theta_stack, r, x, 
                memory, feature_fn, white_params)

    elif objective == 'MCJE':

        S = np.random.normal(loc = 0., scale = 1., 
            size = (theta_stack_init.shape[0], n_draws))

        def objective(theta_stack, grad):
            return path_monte_carlo_entropy_model(theta_stack, r, x, 
                memory, feature_fn, white_params, n_draws = n_draws, S = S)

    elif objective == 'MIE':

        def objective(theta_stack, grad):
            return path_marginalised_entropy_model(theta_stack, r, x, 
                memory, feature_fn, white_params)

    # Define the path constraint
    def constraint(result, theta_stack, grad):
        result = path_bounds_model(theta_stack, r, x, feature_fn.Xq_ref, bound)

    # Obtain the number of parameters involvevd
    n_params = theta_stack_init.shape[0]

    # Prepare for global or local optimisation according to specification
    if globalopt:
        opt = nlopt.opt(nlopt.G_MLSL_LDS, n_params)
        local_opt = nlopt.opt(nlopt.LN_COBYLA , n_params)
        opt.set_local_optimizer(local_opt)
    else:
        opt = nlopt.opt(nlopt.LN_COBYLA , n_params)

    # Set lower and upper bound
    if theta_stack_low is not None:
        theta_stack_low[0] = theta_stack_init[0] - turn_limit
        opt.set_lower_bounds(theta_stack_low)
    if theta_stack_high is not None:
        theta_stack_high[0] = theta_stack_init[0] + turn_limit
        opt.set_upper_bounds(theta_stack_high)

    # Set tolerances
    if xtol_rel > 0:
        opt.set_xtol_rel(xtol_rel)
    if ftol_rel > 0:
        opt.set_ftol_rel(ftol_rel)
    
    # Set maximum optimisation time
    opt.set_maxtime(walltime)

    # Set the objective and constraint and optimise!
    opt.set_max_objective(objective)
    opt.add_inequality_mconstraint(constraint, ctol * np.ones(n_params))
    theta_stack_opt = opt.optimize(theta_stack_init)
    entropy_opt = opt.last_optimum_value()

    # Compute optimal path
    x_abs_opt = forward_path_model(theta_stack_opt, r, x)

    # Return path coordinates, path angles, and path entropy
    return x_abs_opt, theta_stack_opt, entropy_opt

def random_path(theta_stack_init, r, x, memory, feature_fn, white_params, 
    bound = 100, chaos = False):

    # Seed randomly to guarantee no repeats from previous tests
    if chaos:
        np.random.seed(int(time.strftime("%M%S", time.gmtime())))

    # Randomly generate path angles that correspond to a valid path
    n_params = theta_stack_init.shape[0]
    theta_stack = np.random.uniform(0, 2 * np.pi, size = n_params)
    while path_bounds_model(theta_stack, r, x, feature_fn.Xq_ref, bound) > 0:
        theta_stack = np.random.uniform(0, 2 * pi, size = n_params)

    # Compute path coordinates
    x_abs = forward_path_model(theta_stack, r, x)

    # Compute path entropy
    entropy = path_linearised_entropy_model(theta_stack, r, x, 
                    memory, feature_fn, white_params)

    # Return path coordinates, path angles, and path entropy
    return x_abs, theta_stack, entropy

def forward_path_model(theta_stack, r, x):
    """
    Compute path coordinates from path angles, step size, and current location
    """
    theta = np.cumsum(theta_stack)
    x1_rel = np.cumsum(r * np.cos(theta))
    x2_rel = np.cumsum(r * np.sin(theta))
    x_rel = np.array([x1_rel, x2_rel]).T
    return x + x_rel

def path_bounds_model(theta_stack, r, x, Xq_ref, bound):
    """
    Path Constraint
    """
    Xq = forward_path_model(theta_stack, r, x)
    c = cdist(Xq, Xq_ref).min(axis = 1) - bound
    logging.debug('Contraint Violation: {0}'.format(c))
    return c

def path_linearised_entropy_model(theta_stack, r, x, memory, feature_fn, white_params):

    Xq = forward_path_model(theta_stack, r, x)
    Fqw = feature_fn(Xq, white_params)

    logging.info('Computing linearised entropy...')
    start_time = time.clock()

    predictors = gp.classifier.query(memory, Fqw)
    yq_exp = gp.classifier.expectance(memory, predictors)
    yq_cov = gp.classifier.covariance(memory, predictors)
    entropy = gp.classifier.linearised_entropy(yq_exp, yq_cov, memory)

    logging.debug('Linearised entropy computational time: %.8f' % 
        (time.clock() - start_time))
    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_stack), entropy))
    return entropy

def path_monte_carlo_entropy_model(theta_stack, r, x, memory, feature_fn, white_params,
    n_draws = 1000, S = None):

    Xq = forward_path_model(theta_stack, r, x)
    Fqw = feature_fn(Xq, white_params)

    logging.info('Computing monte carlo joint entropy...')
    start_time = time.clock()

    predictors = gp.classifier.query(memory, Fqw)
    yq_exp = gp.classifier.expectance(memory, predictors)
    yq_cov = gp.classifier.covariance(memory, predictors)
    entropy = gp.classifier.monte_carlo_joint_entropy(yq_exp, yq_cov, memory, 
        n_draws = n_draws, S = S)

    logging.debug('Monte carlo joint entropy computational time: %.8f' % 
        (time.clock() - start_time))
    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_stack), entropy))
    return entropy

def path_marginalised_entropy_model(theta_stack, r, x, memory, feature_fn, white_params):

    Xq = forward_path_model(theta_stack, r, x)
    Fqw = feature_fn(Xq, white_params)

    logging.info('Computing marginalised information entropy...')
    start_time = time.clock()

    yq_prob = gp.classifier.predict(Fqw, memory, fusemethod = 'EXCLUSION')
    entropy = gp.classifier.entropy(yq_prob).sum()

    logging.debug('Marginalised information entropy computational time: %.8f' % 
        (time.clock() - start_time))
    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_stack), entropy))
    return entropy

def shift_path(theta_stack, k_step = 1):
    """
    Shift the stacking angles k_steps ahead
    """
    theta_stack_next = np.zeros(theta_stack.shape)
    theta_stack_next[0] = theta_stack[:(k_step + 1)].sum() % (2 * np.pi)
    theta_stack_next[1:-k_step] = theta_stack[(k_step + 1):]
    return theta_stack_next

def correct_lookahead_predictions(xq_abs_opt, learned_classifier, feature_fn, white_params, 
    decision_boundary):
    """
    Obtain the number of steps we can skip optimisation procedures for by 
    looking at the number of correct predictions ahead
    """
    # See how many correct predictions we have
    try:

        # Obtain the unique labels
        if isinstance(learned_classifier, list):
            y_unique = learned_classifier[0].cache.get('y_unique')
        else:
            y_unique = learned_classifier.cache.get('y_unique')

        # Whiten the features
        xqw_abs_opt = feature_fn(xq_abs_opt, white_params)

        # Obtain the predicted classes
        yq_pred = gp.classifier.classify(gp.classifier.predict(xqw_abs_opt, 
                learned_classifier), y_unique)

        # Obtain the true classes
        yq_true = gp.classifier.utils.make_decision(xq_abs_opt, 
            decision_boundary)

        # Obtain the number of correct predictions ahead
        k_step = np.arange(yq_pred.shape[0])[yq_pred != yq_true].argmin() + 1

        # Don't skip too much
        if k_step > yq_pred.shape[0]/2:
            k_step = int(yq_pred.shape[0]/2)

    # If anything goes wrong, perform no skips
    except Exception as e:
        logging.warning(e)
        k_step = 1

    return k_step



