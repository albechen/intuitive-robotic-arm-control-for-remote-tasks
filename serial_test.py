#%%
import serial
import struct
from src.angle_calc.inverse_kinematics import calculate_angles_given_joint_loc
from time import sleep

arduino_serial = serial.Serial("COM5", 9600)
# %%
lens_list = [8, 24.5, 20, 0]

bounds_list = [
    {"min": 0, "max": 180},
    {"min": 0, "max": 135},
    {"min": 0, "max": 135},
    {"min": 0, "max": 180},
    {"min": 0, "max": 180},
    {"min": 0, "max": 180},
]


cordList = [
    [18, 18, 32],
    [24, 15, 20],
    [23, 30, 23],
    [25, 23, 35],
    [30, 14, 24],
]

for cord in cordList:
    print(calculate_angles_given_joint_loc(cord[0], cord[1], cord[2], lens_list))

raw_wrist_angles = [
    [10, 120, 90],
    [50, 100, 70],
    [40, 80, 20],
    [90, 100, 30],
    [30, 130, 60],
]
wrist_bound_list = []

#%%
for cord, wrist in zip(cordList, raw_wrist_angles):
    wrist_1, wrist_2, wrist_3 = wrist
    arm_1, arm_2, arm_3 = calc_arm_angles(cord, lens_list, bounds_list)
    arduino_serial.write(
        struct.pack(">BBBBBB", arm_1, arm_2, arm_3, wrist_1, wrist_2, wrist_3)
    )
    sleep(3)
# %%
