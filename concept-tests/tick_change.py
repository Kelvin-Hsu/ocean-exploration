import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

x = np.linspace(0, 10, num = 20)
y = x**2

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('plot')

ticksize = 30
axis_scale = 10

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(ticksize) 
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(ticksize)

ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/axis_scale))
plt.gca().xaxis.set_major_formatter(ticks)

plt.show()

