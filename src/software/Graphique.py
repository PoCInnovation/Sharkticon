import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
import sys
from threading import Thread
from time import sleep
from tkinter import Button, Label, Tk
from src.software.Sniffer import sniff_packets
from Model.Trainer import train
from Model.Evaluate import predicate, getLastLine
from src.Anomalie_detection import computeAnomalieScore


from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

def proc_sniffer():
    cmd = ['python', 'src/software/Sniffer.py']
    proc = subprocess.Popen(cmd, stdout=sys.stderr.fileno(), stderr=sys.stderr.fileno())
    return proc

class GraphicPage(tk.Frame):

    def __init__(self, parent, controller, font, f):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.label = Label(self, text="Predicting...")
        self.label.pack()
        #self.__filepath = './data/samplefile.txt'
        #self.__fig = plt.figure()
        #self.__ax1 = self.__fig.add_subplot(1, 1, 1)
        #self.__graph = animation.FuncAnimation(self.__fig, self.animate, interval=1000)
        self.__f = f
        self.__canvas = FigureCanvasTkAgg(f, master=self)
        self.action_button = Button(
            self, text="Re-train", fg="red", font=font, command=self.retrain)
        # self.action_button["state"] = tk.DISABLED
        self.__button_quit = Button(
            self, text="Quit", fg="red", font=font, command=self.quit)
        self.__log_label_good = Label(
            self, text='safe', fg="green", font=font)
        self._thread, self._stopTraining, self._stop = None, True, True
        self.bind("<<ShowFrame>>", self.display)
        self.sniffer = None
        self.training_path = "./Model/checkpoints/train"
        self.anomalie = 1
        self.sniffer_stdout = proc_sniffer()
        self.__log_label_good.after(1000, self.change_label)

    def display(self, event) -> None:
        self.__canvas.draw()
        self.__canvas.get_tk_widget().place(relx=0.1, rely=0)
        self.action_button.place(relx=0.15, rely=0.9, anchor=tk.CENTER)
        self.__button_quit.place(relx=0.8, rely=0.82)
        self.__log_label_good.place(relx=0.29, rely=0.85)
        self.label.place(relx=0.15, rely=0.95)
        self.start()
        print("Hello")
        return

    def start(self):
        if self._thread is None:
            self._stop = False
            self._thread = Thread(target=self.run)
            self._thread.start()
        self.label["text"] = "Predicting ..."

    def change_label(self):
        graph_data = open('./data/samplefile.txt', 'r').read()
        lines = graph_data.split('\n')
        i = 1
        my_list = []
        for line in lines[-100:]:
            if len(line) > 0:
                if int(line) > 70:
                    my_list.append(1)
                else:
                    my_list.append(0)
        if my_list.count(1) * 2 > my_list.count(0):
            self.__log_label_good.configure(text="Anomaly detected", fg="red")
        else:
            self.__log_label_good.configure(text="Safe", fg="green")
        self.__log_label_good.after(1000, self.change_label)

    def run(self):
        while True:
            if self._stop:
                break
            if not self._stopTraining:
                self.label["text"] = "Training..."
                train('./src/checkpoints', './data/capture.csv')
                self.label["text"] = "Training finishing.."
                self._stopTraining = True
            self.predict()

    def retrain(self):
        self._stopTraining = False
        self.action_button.configure(text="Cancel", command=self.stop)

    def predict(self):
        self._stopTraining = True
        self.label["text"] = "Predicting..."
        try:
            new_packet = getLastLine('./data/capture.csv'))
            if self.last_packet == new_packet:
                return
            self.last_packet = new_packet
            prediction = predicate(self.training_path, "Execution/Dataset_test.csv", getLastLine('./data/capture.csv'))
            prediction = prediction.numpy().decode("utf-8").split("[SEP]")
            print(prediction)
            with open('./data/samplefile.txt', 'a') as list_scores:
                print()
                list_scores.writelines(computeAnomalieScore(prediction))
        except Exception:
            txt = self.sniffer_stdout.communicate()
            if txt != b'':
                print(txt)
        # TODO: anomaly_Detection
        # sleep(0.1)

    def stop(self):
        self._thread._delete()
        if self._thread is not None:
            self._thread, self._stopTraining, self._stop = None, False, True
        self.label["text"] = "Training aborted"
        isStop = True
        self._stopTraining = True
        print("stop")
        self.action_button.configure(text="Re-Train")

    def quit(self) -> None:
        plt.close(self.__f)
        self.stop()
        self.controller.destroy()
