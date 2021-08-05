import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from threading import Thread
from time import sleep
from tkinter import Button, Label, Tk
from src.software.Sniffer import SharktikonCore
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

class GraphicPage(tk.Frame):

    def __init__(self, parent, controller, font):
        tk.Frame.__init__(self, parent)

        self.__filepath = './data/samplefile.txt'
        self.__fig = plt.figure()
        self.__ax1 = self.__fig.add_subplot(1, 1, 1)
        self.__graph = animation.FuncAnimation(self.__fig, self.animate, interval=1000)
        self.__canvas = FigureCanvasTkAgg(self.__fig, master=self)
        self.__retrain_button = Button(self, text="Re-Train", fg="green", font=font, command=self.retrain)
        self.__button_quit = Button(self, text="Quit", fg="red", font=font, command=self.quit)
        self.__log_label_good = Label(self, text='Anomaly Detected', fg="red", font=font)
        self.sharkticonCore = SharktikonCore()
        self.display()

    def display(self) -> None:
        self.__canvas.draw()
        self.__canvas.get_tk_widget().place(relx=0.1, rely=0)
        self.__retrain_button.place(relx=0.15, rely=0.9, anchor=tk.CENTER)
        self.__button_quit.place(relx=0.8, rely=0.82)
        self.__log_label_good.place(relx=0.29, rely=0.85)
        #self.sharkticonCore.StartSharkticon()
        return

    def animate(self, i):
        graph_data = open(self.__filepath, 'r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        i = 1
        for line in lines:
            if len(line) > 0:
                xs.append(float(i))
                ys.append(float(line))
                i += 1
        self.__ax1.clear()
        self.__ax1.plot(xs, ys)

    def retrain(self):
        print("retrain")

    def quit(self) -> None:
        plt.close(self.__fig)
        self.destroy()

class App(Tk):

    def __init__(self):
        Tk.__init__(self)
        self.label = Label(self, text="Stopped.")
        self.label.pack()
        self.play_button = Button(self, text="Play", command=self.play)
        self.play_button.pack(side="left", padx=2, pady=2)
        self.stop_button = Button(self, text="Stop", command=self.stop)
        self.stop_button.pack(side="left", padx=2, pady=2)
        self._thread, self._pause, self._stop = None, False, True

    def action(self):
        for i in range(1000):
            if self._stop:
                break
            while self._pause:
                self.label["text"] = "Pause... (count: {})".format(i)
                sleep(0.1)
            self.label["text"] = "Playing... (count: {})".format(i)
            sleep(0.1)
        self.label["text"] = "Stopped."

    def play(self):
        if self._thread is None:
            self._stop = False
            self._thread = Thread(target=self.action)
            self._thread.start()
        self._pause = False
        self.play_button.configure(text="Pause", command=self.pause)

    def pause(self):
        self._pause = True
        self.play_button.configure(text="Play", command=self.play)

    def stop(self):
        if self._thread is not None:
            self._thread, self._pause, self._stop = None, False, True
        self.play_button.configure(text="Play", command=self.play)
