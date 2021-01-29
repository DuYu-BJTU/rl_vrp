import socket
import smtplib
import json
from email.mime.text import MIMEText


def send_mail(context=None, is_error=False):
    mail_file = "../../mail_info.json"
    with open(mail_file, "r") as mf:
        mail_info = json.load(mf)
        mail_host = mail_info["mail_host"]
        mail_user = mail_info["mail_user"]
        mail_pass = mail_info["mail_pass"]
        sender = mail_info["sender"]
        receivers = mail_info["receivers"]

    host_name = socket.gethostname()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]

    if is_error:
        message = MIMEText(ip + '\nHi, Training process on ' + host_name +
                           ' has some problems. Just have a see.\n\n' + context,
                           'plain', 'utf-8')
        message['Subject'] = 'Training Problems occur on ' + host_name
    elif context:
        message = MIMEText(ip + '\nHi, Training process on ' + host_name +
                           ' has complete. Just have a see.\n' + context,
                           'plain', 'utf-8')
        message['Subject'] = 'Training Process on ' + host_name + ' Final'
    else:
        message = MIMEText(ip + '\nHi, Training process on ' + host_name + ' has complete. Just have a see.',
                           'plain', 'utf-8')
        message['Subject'] = 'Training Process on ' + host_name + ' Final'

    message['From'] = sender
    message['To'] = receivers[0]

    try:
        smtpObj = smtplib.SMTP(mail_host, 587)
        smtpObj.starttls()
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        smtpObj.quit()
        print('Mail Send Success')

    except smtplib.SMTPException as e:
        print('error', e)

    pass
