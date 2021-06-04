#Transform dataset for Transformers :p

import csv

def readColCSV(fichier, sep, n):
    file = open(fichier, "r")
    reader = csv.reader(file, delimiter = sep)
    col = []
    for row in reader:
        try:
            col.append(row[n])
        except:
            pass
    file.close()
    return col



ipsrc = readColCSV("./dataset/01-12/UDPLag.csv", ",", 2)
portsrc = readColCSV("./dataset/01-12/UDPLag.csv", ",", 3)
ipdest = readColCSV("./dataset/01-12/UDPLag.csv", ",", 4)
portdest =readColCSV("./dataset/01-12/UDPLag.csv", ",", 5)
protocol =readColCSV("./dataset/01-12/UDPLag.csv", ",", 6)

word = {}

for i in range(len(ipsrc)):
    ipsrc



def writeCSV(fichier, sep, ipsrc, portsrc, ipdest, portdest):
    file = open(fichier, "w")
    writer = csv.writer(file, delimiter=sep)
    for i in range(len(ipsrc)):
        writer.writerow((ipsrc[i], portsrc[i], ipdest[i], portdest[i], protocol[i]))
    file.close()

writeCSV("Dataset_final.csv", ",", ipsrc, portsrc, ipdest, portdest)