import socket
import numpy as np
import struct
import time

UDP_IP = "192.168.1.71"
UDP_PORT = 8787

point = np.zeros((10,6),np.float32) # px,py,pz,rx,ry,rz
print(point)

point[0] = [-4.252,319.148,327.137,-180,0,180]
point[1] = [104.103,543.254,-9.818,-180,0,-177.829]
point[2] = [0.0 ,0.0,0.0,0.0,0.0,3.0] # wait 
point[3] = [0.0 ,0.0,0.0,0.0,0.0,1.0] #IO HIGH
point[4] = [0.0 ,0.0,0.0,0.0,0.0,3.0] # wait 
point[5] = [104.104,543.252,56.455,-180,0,180]
point[6] = [0.0 ,0.0,0.0,0.0,0.0,3.0] # wait
point[7] = [30.954,544.270,-9.818,-180,0,180]
point[8] = [0.0 ,0.0,0.0,0.0,0.0,2.0] #IO LOW
point[9] = [-4.252,319.148,327.137,-180,0,180]
# point[3] = [60.954,544.270,-93.255,180,0,180]

# point[0] = [0.0 ,0.0,0.0,0.0,0.0,1.0] #IO HIGH
# point[1] = [-90.63,416.97,308,180,0,180]
# point[2] = [0.0 ,0.0,0.0,0.0,0.0,3.0] # wait 
# point[3] = [-120.0,450.0,308.0,180.0,0.0,180.0]
# point[4] = [0.0 ,0.0,0.0,0.0,0.0,2.0] #IO LOW
# point[2] = [0,0,0,0,0,1]

# point[0] = [-4.252,319.148,327.137,180,0,180] #init
# point[1] = [-104.103,543.254,-9.818,180,0,-177.829] #point
# point[2] = [-104.104,543.252,56.455,180,0,180]#起
# point[3] = [60.954,544.270,-93.255,180,0,180]#

for j in range(10):
    message = b""
    for i in range(6):
        message += struct.pack('f', point[j][i])

    print(message)

    print("UDP target IP: %s" % UDP_IP)
    print("UDP target port: %s" % UDP_PORT)
    print("message: %s" % message)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
    sock.sendto(message, (UDP_IP, UDP_PORT))
    time.sleep(1.5)