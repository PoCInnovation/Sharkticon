import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from threading import Thread
from time import sleep
from tkinter import Button, Label, Tk
from src.software.Sniffer import SharktikonCore
from Model.Trainer import train
from Model.Evaluate import predicate
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

class GraphicPage(tk.Frame):

    def __init__(self, parent, controller, font):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.label = Label(self, text="Stopped.")
        self.label.pack()
        self.__filepath = './data/samplefile.txt'
        self.__fig = plt.figure()
        self.__ax1 = self.__fig.add_subplot(1, 1, 1)
        self.__graph = animation.FuncAnimation(self.__fig, self.animate, interval=1000)
        self.__canvas = FigureCanvasTkAgg(self.__fig, master=self)
        self.retrain_button = Button(self, text="Re-Train", fg="green", font=font, command=self.retrain)
        self.stop_button = Button(self, text="Stop", command=self.stop)
        self.__button_quit = Button(self, text="Quit", fg="red", font=font, command=self.quit)
        self.__log_label_good = Label(self, text='Anomaly Detected', fg="red", font=font)
        self._thread, self._predictRequest, self._stop = None, True, True
        self.sharkticonCore = SharktikonCore()
        self.display()

    def display(self) -> None:
        self.__canvas.draw()
        self.__canvas.get_tk_widget().place(relx=0.1, rely=0)
        self.retrain_button.place(relx=0.15, rely=0.9, anchor=tk.CENTER)
        self.__button_quit.place(relx=0.8, rely=0.82)
        self.__log_label_good.place(relx=0.29, rely=0.85)
        self.label.place(relx=0.15, rely=0.95)
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
        if self._thread is None:
            self._stop = False
            self._thread = Thread(target=self.action)
            self._thread.start()
        self._predictRequest = False
        self.retrain_button.configure(text="Predicting", command=self.predict)

    def action(self):
        if self._stop:
            return
        while self._predictRequest:
            self.label["text"] = "Predicting..."
            #TODO: modifier le retour de la fonction pour qu'il retourne la prediction
            predicate('./data/checkpoints', './data/capture.csv', "GET[SEP]http://localhost:8080/asf-logo-wide.gif~[SEP]HTTP/1.1[SEP]Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)[SEP]no-cache[SEP]no-cache[SEP]text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5[SEP]x-gzip, x-deflate, gzip, deflate[SEP]utf-8, utf-8;q=0.5, *;q=0.5[SEP]en[SEP]localhost:8080[SEP]close[SEP]null[SEP]null[SEP]JSESSIONID=51A7470173188BBB993947F2283059E4[SEP][SEP]anom[SEP]")
            #TODO: anomaly_Detection
            sleep(0.1)
        train('./src/checkpoints', './data/capture.csv')
        self.label["text"] = "Training..."
        sleep(0.1)
        self.label["text"] = "Stopped."

    def predict(self):
        self._predictRequest = True
        self.retrain_button.configure(text="Preditcing", command=self.retrain)

    def stop(self):
        if self._thread is not None:
            self._thread, self._predictRequest, self._stop = None, False, True
        self.stop_button.configure(text="Re-Train", command=self.retrain)

    def quit(self) -> None:
        plt.close(self.__fig)
        self.stop()
        self.controller.destroy()