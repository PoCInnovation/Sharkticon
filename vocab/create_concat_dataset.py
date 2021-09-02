import csv

pathRef = "./dataset_http.csv"
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

print("Begin concatenation . . .")

fp = readColCSV(pathRef, ",", 0)

def writeCSV(fichier, fp):
    file = open(fichier, "w")
    writer = csv.writer(file, delimiter=",")
    writer.writerow(("Packet", "Target"))
    for i in fp:
        i = i.replace("\"", "")
        packetN = i
        nextPacket = i
        writer.writerow((packetN, nextPacket))
    file.close()

writeCSV("dataset_packet_target.csv", fp)

print("Dataset created...")
