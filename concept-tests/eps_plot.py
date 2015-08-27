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
    # 'text.usetex': True,
    'figure.figsize': fig_size(350.0)
};


x = np.linspace(0, 5, 10)
y = x ** 2

# for the publication-ready plots
plt.rc_context(params)

fig, axes = plt.subplots()
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')
fig.savefig('plot.eps')

plt.show()