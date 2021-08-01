from src.software.window import window
import sys

args = sys.argv

if len(args) == 2 and args[1] == "-h":
    print(open("./helper.txt", "r").read())
    exit(0)

app = window('./images/logo.png', '@./images/logo.xbm', 'blue', './data/samplefile.txt')
app.start()