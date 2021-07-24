from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

class window:
    """" handle the management of the window """

    def __init__(self, logo_path: str, ico_path: str, color: str, filepath: str) -> None:
        style.use('fivethirtyeight')
        self.__logo_path = logo_path
        self.__ico_path = ico_path
        self.__window = Tk()
        self.__font = tkFont.Font(family="Arial", size=24, weight="bold")
        self.__title = 'Sharkticon'
        self.__img = Image.open(self.__logo_path)
        self.__img = self.__img.resize((200, 200), Image.ANTIALIAS)
        self.__img = ImageTk.PhotoImage(self.__img)
        self.__logo = Label(self.__window, image=self.__img)
        self.__window.title(self.__title)
        self.__window.iconbitmap(self.__ico_path)
        self.__window.geometry("800x600+550+250")
        self.__start_button = Button(self.__window, text="Start", fg="green", font=self.__font, command=self.display)
        self.__frequency_label = Label(self.__window, text="packets in the graph")
        self.__frequency_choice = ttk.Combobox(self.__window, values=['100', '200', '300', '400'], state='readonly')
        self.__frequency_choice.current(0)
        self.__frequency_choice.bind("<<ComboboxSelected>>", self.action)
        self.__frequency_value = 100
        self.__color = color
        self.__log_var = StringVar()
        self.__filepath = filepath
        self.__fig = plt.figure()
        self.__ax1 = self.__fig.add_subplot(1, 1, 1)
        self.__graph = animation.FuncAnimation(self.__fig, self.animate, interval=1000)
        self.__canvas = FigureCanvasTkAgg(self.__fig, master=self.__window)
        self.__button_quit = Button(self.__window, text="Quit", fg="red", font=self.__font, command=self.quit)
        self.__log_label_good = Label(self.__window, text='No anomaly detected', fg="green", font=self.__font)
        self.home()

    def home(self) -> None:
        self.__start_button.place(relx=0.15, rely=0.70, anchor=CENTER)
        self.__frequency_label.place(relx=0.40, rely=0.68, anchor=CENTER)
        self.__frequency_choice.place(relx=0.40, rely=0.72, anchor=CENTER)
        self.__logo.place(relx=0.49, rely=0.22, anchor=CENTER)

    def display(self) -> None:
        self.__logo.place_forget()
        self.__frequency_label.place_forget()
        self.__frequency_choice.place_forget()
        self.__start_button.place_forget()
        self.__canvas.draw()
        self.__canvas.get_tk_widget().place(relx=0.1, rely=0)
        self.__button_quit.place(relx=0.8, rely=0.82)
        self.__log_label_good.place(relx=0.29, rely=0.85)
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

    def action(self, event) -> None:
        self.__frequency_value = self.__frequency_choice.get()
        print(self.__frequency_value)

    def start(self) -> None:
        self.__window.mainloop()

    def quit(self) -> None:
        plt.close(self.__fig)
        self.__window.destroy()
