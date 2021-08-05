from src.software.window import MainWindow
import sys

args = sys.argv

if len(args) == 2 and args[1] == "-h":
    print(open("./helper.txt", "r").read())
    exit(0)

app = MainWindow('@./images/logo.xbm', 'blue')
app.start()