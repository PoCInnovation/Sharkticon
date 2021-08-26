import json
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def logsManager(self):
    def __init__(self, path: str) -> None:
        self.__path = path
        with open(self.__path, "r").read() as config:
            config = json.load(config)
            self.__from_addr = config['fromAddr']
            self.__password_addr = config['passwordAddr']
            self.__to_addr = config['toAddr']

    def send_mail(self, msg: str) -> None:
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %Hh%Mm%Ss%f")[:-3]
        subject = "Sharkticon anomaly detected"
        body = "Date : " + current_time + "ms\nLog : " + msg + "\n\nUne anomalie nécessite une " \
                                                               "verification\n"
        msg = MIMEMultipart()
        msg['From'] = self.__from_addr
        msg['To'] = self.__to_addr
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        try:
            s.login(self.__from_addr, self.__password_addr)
        except Exception:
            print("Error : your mail config isn't valid")
            return
        text = msg.as_string()
        s.sendmail(self.__from_addr, self.__to_addr, text)
        s.quit()
