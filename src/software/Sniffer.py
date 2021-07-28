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
        self.Path = {"PATH_SAVE": "./data/",
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
        self.fieldnames = ['index', 'method', 'url', 'protocol', 'userAgent', 'pragma', 'cacheControl', 'accept', 'acceptEncoding', 'acceptCharset', 'acceptLanguage', 'host', 'connection', 'contentLength', 'contentType', 'cookie', 'payload', 'label']
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
        try:
            print(packet)
            with open(f"{self.Path['PATH_SAVE']}capture.csv", 'a') as file_data:
                deltatime = time.time() - self.time
                self.time = time.time()
                writer = csv.DictWriter(file_data, self.fieldnames)
                writer.writerow({
                    'index': deltatime,
                    'method': packet.http.request_method,
                    'url': packet.http.request_full_uri,
                    'protocol': packet.http.request.version,
                    'userAgent': packet.http.user_agent,
                    'pragma': '0',
                    'cacheControl': '0',
                    'accept': packet.http.accept,
                    'acceptEncoding': packet.http.accept_encoding,
                    'acceptCharset': packet.http.accept_charset,
                    'acceptLanguage': packet.http.accept_language,
                    'host': packet.http.host,
                    'connection': packet.http.connection,
                    'contentLength': packet.http.content_length,
                    'contentType': packet.http.content_type,
                    'cookie': packet.http.cookie,
                    'payload': packet.http.request_uri,
                    'label': 'anom',
                })
        except AttributeError:
            pass

    def Capturing(self):
        while(1):
            if not self.Status['PROCESS']:
                print(f"Capturing for {self.IA['TIME']} seconds")
                try:
                    capture = pyshark.LiveCapture(interface='wlan0', display_filter='http')
                    print(len(capture))
                    capture.apply_on_packets(self.write_capture, timeout=5)
                    capture.sniff(timeout=3)
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
