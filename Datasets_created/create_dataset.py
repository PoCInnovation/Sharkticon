import csv
import argparse

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

def writeCSV(fichier, sep, cols, len_max):
    file = open(fichier, "w")
    writer = csv.writer(file, delimiter=sep)
    writer.writerow(("Packet", "Target"))
    packetN = ""
    nextPacket = ""
    for i in range(len_max - 1):
        for x in range(len(cols)):
            packetN += cols[x][i] + "[SEP]"
        for x in range(len(cols) - 1):
            nextPacket += cols[x + 1][i] + "[SEP]"
        writer.writerow((packetN, nextPacket))
        packetN = ""
        nextPacket = ""
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Datasets in one string')
    parser.add_argument('path', type=str,
                        help='Path of the dataset to convert')
    parser.add_argument('len_max', type=int,
                        help='Path of the dataset to convert')
    parser.add_argument('name', type=str,
                        help='New name of the dataset created')
    args = parser.parse_args()
    print("Begin creation...")
    cols = ["index", "method", "url", "protocol", "userAgent", "pragma", "cacheControl", "accept", "acceptEncoding",
            "acceptCharset", "acceptLanguage", "host", "connection", "contentLength", "contentType", "cookie", "payload", "label"]
    cols_content = []
    for i in range(1, len(cols)):
        cols_content.append(readColCSV(args.path, ",", i))
    print("Reference dataset loaded...")
    writeCSV("Dataset_" + args.name + ".csv", ",", cols_content, args.len_max)
    print("Dataset created...")
