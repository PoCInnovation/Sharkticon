import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

class graph:
    """ A class for the interactive graph """

    def __init__(self, color, filepath):
        self.__color = color
        self.__filepath = filepath
        self.__fig = plt.figure()
        self.ax1 = fig.add_subplot(1, 1, 1)

    def switch_color(self, new_color):
        self.__color = new_color

    def animate(self):
        graph_data = open('software/data/samplefile.txt', 'r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        i = 1
        for line in lines:
            if len(line) > 0:
                xs.append(float(i))
                ys.append(float(line))
                i += 1
        ax1.clear()
        ax1.plot(xs, ys)





style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

def animate(i):
    graph_data = open('software/data/samplefile.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    i = 1
    for line in lines:
        if len(line) > 0:
            xs.append(float(i))
            ys.append(float(line))
            i += 1
    ax1.clear()
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
