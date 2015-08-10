"""
Receding Horizon Informative Exploration

Path planning methods for receding horizon informative exploration
"""
import numpy as np
import nlopt
from computers import gp
import computers.unsupervised.whitening as pre
import logging
import time

def random_path(theta_stack_init, x, r, memory, whitenparams, ranges, perturb_deg = 30):

    theta_perturb = np.random.normal(loc = 0., scale = np.deg2rad(perturb_deg), 
        size = theta_stack_init.shape[0])

    theta_stack = theta_stack_init + theta_perturb

    while path_bounds_model(theta_stack, r, x, ranges) > 0:

        theta_perturb = np.random.normal(loc = 0., scale = np.deg2rad(perturb_deg), 
            size = theta_stack_init.shape[0])

        theta_stack = theta_stack_init + theta_perturb

    x_abs = forward_path_model(theta_stack, r, x)

    entropy = path_linearised_entropy_model(theta_stack, r, x, 
                    memory, whitenparams)
    return x_abs, theta_stack, entropy

def optimal_path(theta_stack_init, x, r, memory, whitenparams, ranges,
    objective = 'LE', theta_stack_low = None, theta_stack_high = None, 
    walltime = None, xtol_rel = 1e-2, ftol_rel = 1e-2, globalopt = False,
    n_draws = 5000):

    ##### OPTIMISATION #####
    # Propose an optimal path
    try:

        # Select the approach objective
        if objective == 'LE':
            def objective(theta_stack, grad):
                return path_linearised_entropy_model(theta_stack, r, x, 
                    memory, whitenparams)
        elif objective == 'MCJE':
            S = np.random.normal(loc = 0., scale = 1., 
                size = (theta_stack_init.shape[0], n_draws))
            def objective(theta_stack, grad):
                return path_monte_carlo_entropy_model(theta_stack, r, x, 
                    memory, whitenparams, n_draws = n_draws, S = S)
        elif objective == 'MIE':
            def objective(theta_stack, grad):
                return path_marginalised_entropy_model(theta_stack, r, x, 
                    memory, whitenparams)

        # Define the path constraint
        def constraint(theta_stack, grad):
            return path_bounds_model(theta_stack, r, x, ranges)

        # Obtain the number of parameters involvevd
        n_params = theta_stack_init.shape[0]

        # Prepare for global or local optimisation according to specification
        if globalopt:
            opt = nlopt.opt(nlopt.G_MLSL_LDS, n_params)
            local_opt = nlopt.opt(nlopt.LN_COBYLA , n_params)
            opt.set_local_optimizer(local_opt)
        else:
            opt = nlopt.opt(nlopt.LN_COBYLA , n_params)

        # Setup optimiser
        opt.set_lower_bounds(theta_stack_low)
        opt.set_upper_bounds(theta_stack_high)
        opt.set_maxtime(walltime)
        if xtol_rel:
            opt.set_xtol_rel(xtol_rel)
        if ftol_rel:
            opt.set_ftol_rel(ftol_rel)
        
        # Set the objective and constraint and optimise!
        opt.set_max_objective(objective)
        opt.add_inequality_constraint(constraint, 1e-2)
        theta_stack_opt = opt.optimize(theta_stack_init)
        entropy_opt = opt.last_optimum_value()

    # If there is any problem, skip optimisation
    except Exception as e:

        # Note down the error and move the path
        theta_stack_opt = shift_path(theta_stack_init)
        entropy_opt = np.nan
        logging.warning('Problem with optimisation. Continuing planned route.')
        logging.warning(type(e))
        logging.warning(e)
        logging.debug('Initial parameters: {0}'.format(theta_stack_init))

    ##### PATH COMPUTATION #####
    x_abs_opt = forward_path_model(theta_stack_opt, r, x)

    return x_abs_opt, theta_stack_opt, entropy_opt

def forward_path_model(theta_stack, r, x):

    theta = np.cumsum(theta_stack)
    x1_rel = np.cumsum(r * np.cos(theta))
    x2_rel = np.cumsum(r * np.sin(theta))
    x_rel = np.array([x1_rel, x2_rel]).T
    return x + x_rel

def path_linearised_entropy_model(theta_stack, r, x, memory, whitenparams):

    Xq = forward_path_model(theta_stack, r, x)
    Xqw = pre.whiten(Xq, whitenparams)

    logging.info('Computing linearised entropy...')
    start_time = time.clock()

    predictors = gp.classifier.query(memory, Xqw)
    yq_exp = gp.classifier.expectance(memory, predictors)
    yq_cov = gp.classifier.covariance(memory, predictors)
    entropy = gp.classifier.linearised_entropy(yq_exp, yq_cov, memory)

    logging.debug('Linearised entropy computational time: %.8f' % 
        (time.clock() - start_time))
    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_stack), entropy))
    return entropy

def path_monte_carlo_entropy_model(theta_stack, r, x, memory, whitenparams,
    n_draws = 1000, S = None):

    Xq = forward_path_model(theta_stack, r, x)
    Xqw = pre.whiten(Xq, whitenparams)

    logging.info('Computing monte carlo joint entropy...')
    start_time = time.clock()

    predictors = gp.classifier.query(memory, Xqw)
    yq_exp = gp.classifier.expectance(memory, predictors)
    yq_cov = gp.classifier.covariance(memory, predictors)
    entropy = gp.classifier.monte_carlo_joint_entropy(yq_exp, yq_cov, memory, 
        n_draws = n_draws, S = S)

    logging.debug('Monte carlo joint entropy computational time: %.8f' % 
        (time.clock() - start_time))
    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_stack), entropy))
    return entropy

def path_marginalised_entropy_model(theta_stack, r, x, memory, whitenparams):

    Xq = forward_path_model(theta_stack, r, x)
    Xqw = pre.whiten(Xq, whitenparams)

    logging.info('Computing marginalised information entropy...')
    start_time = time.clock()

    yq_prob = gp.classifier.predict(Xqw, memory, fusemethod = 'EXCLUSION')
    entropy = gp.classifier.entropy(yq_prob).sum()


    logging.debug('Marginalised information entropy computational time: %.8f' % 
        (time.clock() - start_time))
    logging.debug('Angles (deg): {0} | Entropy: {1}'.format(
        np.rad2deg(theta_stack), entropy))
    return entropy

def path_bounds_model(theta_stack, r, x, ranges):
    """
    Path Constraint

    This assumes that the field is a square about the origin
    """
    Xq = forward_path_model(theta_stack, r, x)
    c = np.max(np.abs(Xq)) - ranges[1]
    logging.debug('Contraint Violation: %.5f' % c)
    return c

def correct_lookahead_predictions(xq_abs_opt, learned_classifier, whitenparams, 
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
        xqw_abs_opt = pre.whiten(xq_abs_opt, whitenparams)

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

def shift_path(theta_stack, k_step = 1):
    """
    Shift the stacking angles k_steps ahead
    """
    theta_stack_next = np.zeros(theta_stack.shape)
    theta_stack_next[0] = theta_stack[:(k_step + 1)].sum() % (2 * np.pi)
    theta_stack_next[1:-k_step] = theta_stack[(k_step + 1):]
    return theta_stack_next