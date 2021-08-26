from tkinter import ttk
import tkinter.font as tkFont
from tkinter import messagebox
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from src.software.Sniffer import SharktikonCore
from src.software.Graphique import GraphicPage
from src.software.startingPage import StartingPage
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
import tkinter as tk

# sharkticonCore = SharktikonCore()

class MainWindow(tk.Tk):
    """" handle the management of the MainWindow """

    def __init__(self, ico_path: str, color: str):
        tk.Tk.__init__(self)
        style.use('fivethirtyeight')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.__ico_path = ico_path
        self.font = tkFont.Font(family="Arial", size=24, weight="bold")
        self.__title = 'Sharkticon'
        self.title(self.__title)
        # self.iconbitmap(self.__ico_path)
        self.geometry("800x600+550+250")
        self.__color = color
        self.__log_var = tk.StringVar()
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for F in (StartingPage, GraphicPage):
            frame = F(container, self, self.font, f)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartingPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        frame.update()
        frame.event_generate("<<ShowFrame>>")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()

    def start(self) -> None:
        self.mainloop()


f = plt.figure()
ax1 = f.add_subplot(1, 1, 1)


def animate(i):
    graph_data = open('./data/samplefile.txt', 'r').read()
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


app = MainWindow('@./images/logo.xbm', 'blue')
ani = animation.FuncAnimation(f, animate, interval=1000)
app.start()
