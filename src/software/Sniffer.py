from scapy.all import *
from scapy.layers.http import HTTPRequest, HTTPResponse
from colorama import init, Fore

init()
# define colors
GREEN = Fore.GREEN
RED   = Fore.RED
RESET = Fore.RESET


def sniff_packets():
    sniff(prn=process_packet, iface="wlan0", store=False)


def process_packet(packet):
    """
    This function is executed whenever a packet is sniffed
    """
    if packet.haslayer(HTTPRequest):
        print("New packet")
        with open("./data/capture.csv", 'a') as file_data:
            attributs = [packet[HTTPRequest].Method.decode(),
                         packet[IP].src,
                         packet[HTTPRequest].Host.decode() + packet[HTTPRequest].Path.decode(),
                         packet[HTTPRequest].Host.decode() + packet[HTTPRequest].Path.decode(),
                         packet[HTTPRequest].User_Agent.decode(),
                         packet[HTTPRequest].Host.decode(),
                         str(packet[HTTPRequest].Content_Length),
                         str(packet[HTTPRequest].Content_Type)
                         ]
            line = ""
            for i in attributs:
                line += i + "[SEP]"
            line += "\n"
            #print("lIne:", line)
            file_data.writelines(line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HTTP Packet Sniffer, this is useful when you're a man in the middle." \
                                                 + "It is suggested that you run arp spoof before you use this script, otherwise it'll sniff your personal packets")
    parser.add_argument("-i", "--iface", help="Interface to use, default is scapy's default interface")
    parser.add_argument("--show-raw", dest="show_raw", action="store_true", help="Whether to print POST raw data, such as passwords, search queries, etc.")
    # parse arguments
    args = parser.parse_args()
    iface = args.iface
    show_raw = args.show_raw
    sniff_packets()