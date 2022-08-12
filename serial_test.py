# %%
import numpy as np
import serial
import struct
from src.angle_calc.inverse_kinematics import calculate_angles_given_joint_loc
from time import sleep


#%%

arduino_serial = serial.Serial("COM5", 9600)

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
