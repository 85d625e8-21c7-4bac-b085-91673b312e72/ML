import datetime
import socket
import os

print ("Hello from dataset updater!!!")
print("datetime now =", datetime.datetime.now())
print("Hostname =", socket.gethostname())
print("Home directory =", os.path.expanduser('~'))