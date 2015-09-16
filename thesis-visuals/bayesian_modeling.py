import numpy as np
import matplotlib.pyplot as plt
import computers.gp as gp
import computers.unsupervised.whitening as pre
from matplotlib import cm

def kerneldef(h, k):
    a = h(0.1, 5, 0.1)
    b = h(0.1, 5, 0.1)
    return a * k('gaussian', b)

def phenomenon(x, noise):
    f = np.sin(2 * np.pi * x)
    if noise > 0:
        e = np.random.normal(loc = 0.0, scale = noise, size = (x.shape[0], 1))
    else:
        e = 0
    return (f + e)[:, 0]

def center(y, ym = None):

    if ym is None:
        ym = y.mean()
        return y - ym, ym
    else:
        return y + ym

def main(n_train = 0):

    np.random.seed(10)
    n_query = 100
    n_draws = 10
    n_dims = 1

    noise = 0.00

    xl = 0.0
    xu = 1.0

    # Test Data Set
    X = np.random.uniform(xl, xu, size = (n_train, n_dims))
    X = X[np.argsort(X[:,0])]

    if n_train > 0:
        X = np.array([[0.2, 0.8]]).T
    y = phenomenon(X, noise)
    
    Xq = np.linspace(xl, xu, n_query)[:, np.newaxis]

    # We can automatically extract the upper and lower theta vectors
    kernel = gp.compose(kerneldef)
    print_fn = gp.describer(kerneldef)

    if n_train > 0:

        # Whiten inputs and de-mean outputs:
        Xw, whiteparams = pre.whiten(X)
        Xqw = pre.whiten(Xq, whiteparams)
        yw, ym = center(y)

        # Set up optimisation
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(kerneldef)
        opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
        opt_config.walltime = 3.0

        # Learning signal and noise hyperparameters
        hyper_params = gp.learn(Xw, yw, kernel, opt_config)
        print('Final kernel:', print_fn(hyper_params), '+ noise', hyper_params[1])

        regressor = gp.condition(Xw, yw, kernel, hyper_params)
        query = gp.query(Xqw, regressor)

        fq_exp = center(gp.mean(regressor, query), ym)
        fq_cov = gp.covariance(regressor, query)

    else:

        fq_exp = np.zeros(Xq.shape[0])
        fq_cov = 2 * kernel(Xq, Xq, (np.array([0.5, 0.3])))

    visualise(X, y, Xq, fq_exp, fq_cov, noise = noise, n_draws = n_draws)
    return

def visualise(X, y, Xq, fq_exp, fq_cov, noise = 1.0, n_draws = 10):

    FONTSIZE = 25
    FONTNAME = 'Sans Serif'
    TICKSIZE = 24

    mycmap = cm.get_cmap(name = 'gist_rainbow', lut = None)

    rcparams = {
        'backend': 'pdf',
        'axes.labelsize': TICKSIZE,
        'text.fontsize': FONTSIZE,
        'legend.fontsize': FONTSIZE,
        'xtick.labelsize': TICKSIZE,
        'ytick.labelsize': TICKSIZE,
        'text.usetex': True,
        'axes.color_cycle': [mycmap(k) for k in np.linspace(0, 1, n_draws)]
    }

    plt.rc_context(rcparams)

    fq_var = fq_cov.diagonal()
    yq_draws = gp.draws(n_draws, fq_exp, fq_cov)

    yq_var = fq_var + noise**2
    yq_ub = fq_exp + 2 * yq_var
    yq_lb = fq_exp - 2 * yq_var

    if y.shape[0] == 0:
        title = 'prior'
    else:
        title = 'posterior'

    # Plot
    fig1 = plt.figure(figsize = (8.0, 6.0))
    fig2 = plt.figure(figsize = (8.0, 6.0))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    ax1.plot(Xq, fq_exp, 'k-', linewidth = 2)
    ax1.fill_between(Xq[:, 0], yq_ub, yq_lb, facecolor = (0.9, 0.9, 0.9), edgecolor = (0.5, 0.5, 0.5))
    ax1.plot(X[:, 0], y, 'c.')
    ax1.set_ylim((-3.0, 3.0))
    ax1.set_xlabel('$x$', fontsize = FONTSIZE)
    ax1.set_ylabel('$f(x)$', fontsize = FONTSIZE)
    ax1.set_title(title, fontsize = FONTSIZE)

    ax2.plot(Xq, fq_exp, 'k-', linewidth = 2)
    ax2.fill_between(Xq[:, 0], yq_ub, yq_lb, facecolor = (0.9, 0.9, 0.9), edgecolor = (0.5, 0.5, 0.5))
    ax2.plot(X[:, 0], y, 'c.')
    for i in range(n_draws):
        ax2.plot(Xq[:, 0], yq_draws[i], '--')
    ax2.set_ylim((-3.0, 3.0))
    ax2.set_xlabel('$x$', fontsize = FONTSIZE)
    ax2.set_ylabel('$f(x)$', fontsize = FONTSIZE)
    ax2.set_title(title, fontsize = FONTSIZE)

    fig1.tight_layout()
    fig2.tight_layout()

    if y.shape[0] == 0:
        fig1.savefig('bayesian_modeling/prior.eps')
        fig2.savefig('bayesian_modeling/prior_draws.eps')
    else:
        fig1.savefig('bayesian_modeling/posterior%d.eps' % y.shape[0])
        fig2.savefig('bayesian_modeling/posterior_draws%d.eps' % y.shape[0])


if __name__ == "__main__":
    main(n_train = 0)
    main(n_train = 2)
    plt.show()