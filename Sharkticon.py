import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import sys

args = sys.argv

if len(args) == 2 and args[1] == "-h":
    print(open("./helper.txt", "r").read())
    exit(0)

from src.software.window import MainWindow
from src.software.Sniffer