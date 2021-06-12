import csv

def readColCSV(fichier, sep):
    file = open(fichier, "r")
    reader = csv.reader(file)
    lines = []
    for row in reader:
        try:
            lines.append(row)
        except Exception:
            pass
    file.close()
    return lines

lines = readColCSV("./light_dataset.csv", ":")

def writeTXT(fichier, lines):
    file = open(fichier, "w")
    with open(fichier, "w") as file:
        for line in lines:
            line = str(line)[2:-2]
            file.write(line + "\n")
        file.close()

writeTXT("Dataset_shape_1.txt", lines)
