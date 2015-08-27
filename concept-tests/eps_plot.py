import numpy as np
import matplotlib.pyplot as plt

def fig_size(fig_width_pt):
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt# width in inches
    fig_height = fig_width * golden_mean    # height in inches
    return fig_width, fig_height

params = {
    'backend': 'pdf',
    'axes.labelsize': 10,
    'text.fontsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True,
    'figure.figsize': fig_size(350.0)
};


x = np.linspace(0, 5, 10)
y = x ** 2
z = x ** 3

# for the publication-ready plots
plt.rc_context(params)

fig = plt.figure()
axes1 = fig.add_subplot(211)
axes2 = fig.add_subplot(212)

axes1.plot(x, y, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')

axes2.plot(x, z, 'b')
axes2.set_xlabel('x')
axes2.set_ylabel('z')
axes2.set_title('title')

fig.savefig('plot.eps')

plt.show()