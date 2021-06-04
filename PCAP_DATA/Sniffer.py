from threading import Thread
import pandas as pd
import numpy as np
from datetime import datetime
import pyshark
import csv


import os
import signal
import time
import re


class Sharktikon():
    def __init__(self):
        self.Path = {"PATH_SAVE": "./save_capture/",
                     "PATH_PREPROCESS_SCRIPT": "./Sharkticon/",
                     "PATH_MODEL": '.h5'}

        self.Status = {"CAPTURE": True,
                       "PROCESS": False,
                       "STOP": False,
                       "NEW": False,
                       "GO": False,
                       "SAVING": False,
                       "DDOS": True,
                       "MITM": True}

        self.IA = {"MODEL": 0,
                   "PACKETS": [],
                   "NUMBER_BAD_PACKETS": 0,
                   "NUMBER_PACKETS": 0,
                   "PREDICTION": 0,
                   "PACKETS_FLOW": 20,
                   "TIME": 3,
                   "SPY": 0}

        self.time = time.time()
        self.fieldnames = ['deltatime', 'protocol', 'ip_src', 'srcport', 'ip_dst', 'dstport', 'length']
        os.makedirs(self.Path['PATH_SAVE'], exist_ok=True)
        with open(f"{self.Path['PATH_SAVE']}capture.csv", 'a') as file_data:
            writer = csv.DictWriter(file_data, self.fieldnames)
            writer.writeheader()

    def StartSharkticon(self):
        self.CapturingThread = Thread(target=self.Capturing)
        self.ProcessingThread = Thread(target=self.Processing)
        self.CapturingThread.daemon = True
        self.ProcessingThread.daemon = True

        self.CapturingThread.start()
        self.ProcessingThread.start()
        self.CapturingThread.join()
        self.ProcessingThread.join()

    def write_capture(self, packet):
        localtime = time.time()
        #TODO: deltatime
        try:
            with open(f"{self.Path['PATH_SAVE']}capture.csv", 'a') as file_data:
                deltatime = time.time() - self.time
                self.time = time.time()
                writer = csv.DictWriter(file_data, self.fieldnames)
                protocol = packet.transport_layer   # protocol type
                ip_src = packet.ip.src            # source address
                srcport = packet[protocol].srcport   # source port
                ip_dst = packet.ip.dst            # destination address
                dstport = packet[protocol].dstport   # destination port
                length = packet[protocol].length
                writer.writerow({
                                'deltatime': deltatime,
                                'protocol': protocol,
                                'ip_src': ip_src,
                                'srcport': srcport,
                                'ip_dst': ip_dst,
                                'dstport': dstport,
                                'length': length
                                })
        except AttributeError:
            pass

    def Capturing(self):
        while(1):
            if not self.Status['PROCESS']:
                print(f"Capturing for {self.IA['TIME']} seconds")
                try:
                    capture = pyshark.LiveCapture(interface='wlan0')
                    capture.apply_on_packets(self.write_capture, timeout=5)
                    capture.sniff(timeout=3)
                    print(capture)
                    time.sleep(self.IA['TIME'])
                except Exception as e:
                    print(e)
                self.Status['PROCESS'] = True
                time.sleep(1)

    def Processing(self):
        while(1):
            if (self.Status['PROCESS']):
                print("processing...")
                time.sleep(1)
                self.Status['PROCESS'] = False


if __name__ == "__main__":
    Sharktikon().StartSharkticon()
