import csv

pathRef = "./light_dataset.csv"
lenMax = 100

def readColCSV(fichier, sep, n):
    file = open(fichier, "r")
    reader = csv.reader(file, delimiter=sep)
    col = []
    for row in reader:
        try:
            col.append(row[n])
        except Exception:
            pass
    file.close()
    return col

print("Begin creation...")

ipsrc = readColCSV(pathRef, ";", 0)
portsrc = readColCSV(pathRef, ";", 1)
ipdest = readColCSV(pathRef, ";", 2)
portdest = readColCSV(pathRef, ";", 3)
#protocol = readColCSV(pathRef, ";", 4)

#lenMax = len(ipsrc)

print("Reference dataset loaded...")

def writeCSV(fichier, sep, ipsrc, portsrc, ipdest, portdest):
    file = open(fichier, "w")
    writer = csv.writer(file, delimiter=sep)
    writer.writerow(("Packet", "Target"))
    for i in range(lenMax - 1):
        packetN = ipsrc[i] + "[SEP]" + portsrc[i] + "[SEP]" + ipdest[i] + "[SEP]" + portdest[i] + "[SEP]"
        nextPacket = ipsrc[i+1] + "[SEP]" + portsrc[i+1] + "[SEP]" + ipdest[i+1] + "[SEP]" + portdest[i+1] + "[SEP]"
        writer.writerow((packetN, nextPacket))
    file.close()

writeCSV("Dataset_2_cols.csv", ",", ipsrc, portsrc, ipdest, portdest)

print("Dataset created...")