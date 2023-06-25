#%%
import time
from pySerialTransfer import pySerialTransfer as txfer


link = txfer.SerialTransfer('COM5')

link.open()
time.sleep(2) # allow some time for the Arduino to completely reset

while True:
    send_size = 0
    
    ###################################################################
    # Send a list
    ###################################################################
    list_ = [1, 3]
    list_size = link.tx_obj(list_)
    send_size += list_size

    ###################################################################
    # Transmit all the data to send in a single packet
    ###################################################################
    link.send(send_size)
    
    ###################################################################
    # Wait for a response and report any errors while receiving packets
    ###################################################################
    while not link.available():
        if link.status < 0:
            if link.status == txfer.CRC_ERROR:
                print('ERROR: CRC_ERROR')
            elif link.status == txfer.PAYLOAD_ERROR:
                print('ERROR: PAYLOAD_ERROR')
            elif link.status == txfer.STOP_BYTE_ERROR:
                print('ERROR: STOP_BYTE_ERROR')
            else:
                print('ERROR: {}'.format(link.status))
    
    ###################################################################
    # Parse response list
    ###################################################################
    rec_list_  = link.rx_obj(obj_type=type(list_),
                                obj_byte_size=list_size,
                                list_format='i')
    
    ###################################################################
    # Display the received data
    ###################################################################
    # print('SENT: {}'.format(list_, ))
    # print('RCVD: {}'.format(rec_list_))
    print(' ')

# %%
import numpy as np
import serial

from src.angle_calc.inverse_kinematics import calculate_angles_given_joint_loc

#%%
import serial
from struct import pack
import sys
import time
import random
import ast


try:
    ser = serial.Serial(baudrate="115200", timeout=0.5, port="com6")
except:
    print("Port open error")

time.sleep(1)  # no delete!
for x in range(1000):
    ser.write(
        pack(
            "15h",
            -2933,
            94,
            1000,
            29,
            -2000,
            666,
            6,
            7,
            444,
            9,
            10,
            2222,
            12,
            13,
            random.randint(0, 100),
        )
    )  # the 15h is 15 element, and h is an int type data
    # random test, that whether data is updated
    time.sleep(0.01)  # delay
    dat = ser.readline()  # read a line data

    if dat != b"" and dat != b"\r\n":
        try:  # convert in list type the readed data
            dats = str(dat)
            dat1 = dats.replace("b", "")
            dat2 = dat1.replace("'", "")
            dat3 = dat2[:-4]
            list_ = ast.literal_eval(dat3)  # list_ value can you use in program
            print(str(x) + dats)
        except:
            print("Error in corvert, readed: ", dats)
    time.sleep(0.05)
#%%
import serial
import struct
from time import sleep

ser = serial.Serial(baudrate="9600", timeout=0.5, port="COM5")
sleep(10)
#%%
left_angles = [-2933, 94, 1000, 29, -2000]
# left_angles = [1, 2, 3, 4, 5]

for x in range(100):

    sleep(0.1)
    # strlist = [str(x) for x in left_angles]
    # str_ang = "$".join(strlist) + "$"
    # arduino_serial.write(str_ang.encode())
    ser.write(
        struct.pack(
            "5h",
            left_angles[0],
            left_angles[1],
            left_angles[2],
            left_angles[3],
            left_angles[4],
        )
    )

    print(left_angles)

    sleep(0.01)  # delay

    cc = str(ser.readline())
    print("arduino", str(cc))

    left_angles = [x + 1 for x in left_angles]
    sleep(1)
#%%
#Arduino forum 2020 - https://forum.arduino.cc/index.php?topic=714968 
import serial
from struct import *
import sys
import time
import random
import ast


try:
    ser=serial.Serial(baudrate='115200', timeout=.5, port='com5')
except:
    print('Port open error')

time.sleep(5)#no delete!
while True:
    try:
        time.sleep(1)
        ser.write(pack ('15h',0,1,2,3,4,666,6,7,444,9,10,2222,12,13,random.randint(0,100)))#the 15h is 15 element, and h is an int type data
                                                                    #random test, that whether data is updated
        time.sleep(.01)#delay
        dat=ser.readline()#read a line data
        
        if dat!=b''and dat!=b'\r\n':
            try:                #convert in list type the readed data
                dats=str(dat)
                dat1=dats.replace("b","")
                dat2=dat1.replace("'",'')
                dat3=dat2[:-4]
                list_=ast.literal_eval(dat3) #list_ value can you use in program
                print(dat3)
            except:
                print('Error in corvert, readed: ', dats)
        time.sleep(.05)
    except KeyboardInterrupt:
        break
    except:
        print(str(sys.exc_info())) #print error
        break

#the delays need, that the bytes are good order
#%%
arduino_serial.write(
    struct.pack(
        ">BBBBBBB",
        10,
        10,
        10,
        10,
        10,
        10,
        10,
    )
)
# %%
lens_list = [8, 24.5, 23, 3, 5, 2]

bounds_list = [
    {"min": 0, "max": 180},
    {"min": 0, "max": 135},
    {"min": 0, "max": 135},
    {"min": 0, "max": 180},
    {"min": 0, "max": 180},
    {"min": 0, "max": 180},
]

wrist_list = [
    [15, 0, 20],  # Close 0 mid
    [25, 25, 15],  # Far 45 low
    [24, 15, 30],  # Mid 75 high
    [0, 26, 15],  # mid 90 low
]

pointer_pinky_list = [
    [[0, 1, 0], [-1, 1, 0]],  # 90 tilt none face floor
    [[1, 1, -1], [1, 1, -2]],  # 45 tilt down face right
    [[1, 2, 1], [0, 2, 1]],  # 75 tilt up face floor
    [[1, 0, 0], [1, 0, -1]],  # 0 tilt none face me
]

combo_cords = []
for n in range(len(wrist_list)):
    wrist_cord = np.array(wrist_list[n])
    adj_pp_list = [np.add(x, wrist_cord) for x in pointer_pinky_list[n]]
    combo_cords.append([wrist_cord] + (adj_pp_list))

combo_cords

#%%
adj_angle_list = []
raw_angle_list = []
for x in combo_cords:
    raw_angles, adj_angles, end_rot_matrix = calculate_angles_given_joint_loc(
        x[0], x[2], x[1], lens_list, bounds_list
    )
    adj_angle_list.append(adj_angles)
    raw_angle_list.append(raw_angles)
print(np.array(adj_angle_list))
print(np.array(raw_angle_list))

# %%
arduino_serial = serial.Serial("COM5", 9600)

#%%
n = 0
adj_angles = adj_angle_list[n]
print(adj_angles)
arduino_serial.write(
    struct.pack(
        ">BBBBBBB",
        adj_angles[0],
        adj_angles[1],
        adj_angles[2],
        adj_angles[3],
        adj_angles[4],
        adj_angles[5],
        0,
    )
)
#%%
arduino_serial.write(
    struct.pack(
        ">BBBBBBB",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
)
#%%
arduino_serial.write(struct.pack(">BBBBBBB", 90, 90, 90, 90, 90, 90, 120))
#%%
sleep(3)
arduino_serial.write(struct.pack(">BBBBBBB", 90, 90, 90, 90, 90, 90, 120))
sleep(2)
arduino_serial.write(struct.pack(">BBBBBBB", 90, 90, 90, 90, 90, 90, 30))
sleep(2)
arduino_serial.write(struct.pack(">BBBBBBB", 90, 90, 90, 90, 90, 90, 120))
sleep(2)
arduino_serial.write(struct.pack(">BBBBBBB", 90, 90, 90, 90, 90, 90, 30))
sleep(2)
arduino_serial.write(struct.pack(">BBBBBBB", 90, 90, 90, 90, 90, 90, 120))
sleep(2)
arduino_serial.write(struct.pack(">BBBBBBB", 90, 90, 90, 90, 90, 90, 30))
sleep(2)
# %%
for adj_angles in adj_angle_list:
    print(adj_angles)
    arduino_serial.write(
        struct.pack(
            ">BBBBBB",
            adj_angles[0],
            adj_angles[1],
            adj_angles[2],
            adj_angles[3],
            adj_angles[4],
            adj_angles[5],
        )
    )
    sleep(3)
# %%
