import tkinter as tk
from src.software.Graphique import GraphicPage
from PIL import Image, ImageTk


class StartingPage(tk.Frame):

    def __init__(self, parent, controller, font, f):
        tk.Frame.__init__(self, parent)

        self.__start_button = tk.Button(self, text="Start", fg="green", font=font,
                                        command=lambda: controller.show_frame(GraphicPage))
        self.__start_button.pack()
        self.__logo_path = './images/logo.png'
        self.__frequency_label = tk.Label(self, text="packets in the graph")
        self.__frequency_choice = tk.ttk.Combobox(
            self, values=['100', '200', '300', '400'], state='readonly')
        self.__frequency_choice.current(0)
        self.__frequency_choice.bind("<<ComboboxSelected>>", self.action)
        self.__frequency_value = 100
        self.__img = Image.open(self.__logo_path)
        self.__img = self.__img.resize((200, 200), Image.ANTIALIAS)
        self.__img = ImageTk.PhotoImage(self.__img)
        #self.__logo = tk.Label(self, image=self.__img)
        self.draw()

    def draw(self):
        self.__start_button.place(relx=0.15, rely=0.70, anchor=tk.CENTER)
        self.__frequency_label.place(relx=0.40, rely=0.68, anchor=tk.CENTER)
        self.__frequency_choice.place(relx=0.40, rely=0.72, anchor=tk.CENTER)
        #self.__logo.place(relx=0.49, rely=0.22, anchor=tk.CENTER)

    def action(self, event) -> None:
        self.__frequency_value = self.__frequency_choice.get()
        return self.__frequency_value
