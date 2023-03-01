#%%
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


def law_of_cos(e1: float, e2: float, opp: float) -> float:
    ratio = (e1**2 + e2**2 - opp**2) / (2 * e1 * e2)
    opp_angle = np.arccos(ratio)
    return opp_angle


def calc_arm_angles(END_CORD: list, lens: list) -> list:
    X, Y, Z = END_CORD[0], END_CORD[1], END_CORD[2]

    if X == 0 and Y == 0:
        arm_angles = [-999, -999, -999]

    else:
        theta_1 = np.arcsin(Y / (X**2 + Y**2) ** 0.5)
        z_adj = Z - lens[0]
        d = sum([X**2, Y**2, z_adj**2]) ** 0.5

        theta_21 = np.arctan(z_adj / (X**2 + Y**2) ** 0.5)

        if d >= lens[1] + lens[2] + lens[3]:
            theta_2 = theta_21
            theta_3 = 0

        else:
            theta_22 = law_of_cos(lens[1], d, lens[2] + lens[3])
            theta_2 = theta_21 + theta_22

            theta_3 = np.radians(180) - law_of_cos(lens[1], lens[2] + lens[3], d)

        arm_angles = [theta_1, theta_2, theta_3]

    arm_angles = [np.degrees(x) for x in arm_angles]

    return arm_angles


def calc_series_rotation(
    homo_matrix_list: list, first_value: int, last_value: int
) -> np.ndarray:
    first_matrix = True
    overall_rot_matrix = np.array([[], []])
    homo_matrix_slice = homo_matrix_list[first_value:last_value]

    for homo_matrix in homo_matrix_slice:

        if first_matrix == True:
            overall_rot_matrix = homo_matrix
            first_matrix = False
        else:
            overall_rot_matrix = np.dot(overall_rot_matrix, homo_matrix)

    return overall_rot_matrix


def calc_homo_matrix(angles: list, DH_table: list) -> list:

    num_joints = len(angles)
    homo_matrix_list = [np.array(0)] * num_joints

    for n in range(num_joints):
        theta, alpha, r, d = DH_table[n]
        theta = np.radians(theta)
        alpha = np.radians(alpha)

        c_theta = round(np.cos(theta), 10)
        s_theta = round(np.sin(theta), 10)
        c_alpha = round(np.cos(alpha), 10)
        s_alpha = round(np.sin(alpha), 10)

        homo_matrix = [
            [
                c_theta,
                -s_theta * c_alpha,
                s_theta * s_alpha,
                r * c_theta,
            ],
            [
                s_theta,
                c_theta * c_alpha,
                -c_theta * s_alpha,
                r * s_theta,
            ],
            [0, s_alpha, c_alpha, d],
            [0, 0, 0, 1],
        ]
        homo_matrix_list[n] = np.array(homo_matrix)
    return homo_matrix_list


# def return_dh_table(angles, lens):
#     DH_table = [
#         [angles[0] + 0, 90, 0, lens[0]],
#         [angles[1] + 0, 180, lens[1], 0],
#         [angles[2] - 90, -90, 0, 0],
#         [angles[3] + 0, 90, 0, lens[2] + lens[3]],
#         [angles[4] + 90, 90, 0, 0],
#         [angles[5] + 0, 0, 0, lens[4] + lens[5]],
#     ]
#     return DH_table


def return_dh_table(angles: list, lens: list) -> list:
    DH_table = [
        [angles[0] + 0, 90, 0, lens[0]],
        [angles[1] + 0, 180, lens[1], 0],
        [angles[2] - 90, -90, 0, 0],
        [angles[3] + 0, -90, 0, lens[2] + lens[3]],
        [angles[4] + 0, 90, 0, 0],
        [angles[5] + 0, 0, 0, lens[4] + lens[5]],
    ]
    return DH_table


def calc_wirst_rot_matrix(wrist_angles: list) -> np.ndarray:
    wrist_angles_rad = [np.radians(x) for x in wrist_angles]
    agl_z1, agl_y, agl_z12 = wrist_angles_rad

    rot_matrix = np.array(
        [
            [
                np.cos(agl_z1) * np.cos(agl_y) * np.cos(agl_z12)
                - np.sin(agl_z1) * np.sin(agl_z12),
                -np.cos(agl_z1) * np.cos(agl_y) * np.sin(agl_z12)
                - np.sin(agl_z1) * np.cos(agl_z12),
                np.cos(agl_z1) * np.sin(agl_y),
            ],
            [
                np.sin(agl_z1) * np.cos(agl_y) * np.cos(agl_z12)
                + np.cos(agl_z1) * np.sin(agl_z12),
                -np.sin(agl_z1) * np.cos(agl_y) * np.sin(agl_z12)
                + np.cos(agl_z1) * np.cos(agl_z12),
                np.sin(agl_z1) * np.sin(agl_y),
            ],
            [
                -np.sin(agl_y) * np.cos(agl_z12),
                np.sin(agl_y) * np.sin(agl_z12),
                np.cos(agl_y),
            ],
        ]
    )
    return rot_matrix


# def calc_wrist_angles(arm_angles, lens, rot_06):
#     DH_table = return_dh_table(arm_angles, lens)

#     homo_matrix_list = calc_homo_matrix(arm_angles, DH_table)

#     rot_03 = calc_series_rotation(homo_matrix_list, 0, 3)[:3, :3]
#     inv_rot_03 = np.linalg.inv(rot_03)
#     rot_36 = np.dot(inv_rot_03, rot_06)

#     theta_5 = np.arcsin(rot_36[2, 2])
#     if round(theta_5, 10) != round(np.pi / 2, 10):

#         if rot_36[0, 2] == 0:
#             theta_4 = np.pi / 2
#         else:
#             theta_4 = np.arctan(rot_36[1, 2] / rot_36[0, 2])

#         if rot_36[2, 0] == 0:
#             theta_6 = np.pi / 2
#         else:
#             theta_6 = np.arctan(-rot_36[2, 1] / rot_36[2, 0])

#     else:
#         if rot_36[0, 0] == 0:
#             theta_6 = 0
#             theta_4 = 0
#         else:
#             half_angle = np.arctan(rot_36[1, 0] / rot_36[0, 0]) / 2
#             theta_6 = half_angle
#             theta_4 = half_angle

#     return [np.degrees(theta_4), np.degrees(theta_5), np.degrees(theta_6)]


def calc_wrist_angles(arm_angles: list, lens: list, rot_06: list) -> list:
    # https://www.mecharithm.com/explicit-representations-for-the-orientation-euler-angles/
    DH_table = return_dh_table(arm_angles, lens)

    homo_matrix_list = calc_homo_matrix(arm_angles, DH_table)

    rot_03 = calc_series_rotation(homo_matrix_list, 0, 3)[:3, :3]
    inv_rot_03 = np.linalg.inv(rot_03)
    rot_36 = np.dot(inv_rot_03, rot_06)

    if np.sin(np.arccos(rot_36[2, 2])) != 0:
        theta_5 = -np.arctan2(
            -((rot_36[0, 2] ** 2 + rot_36[1, 2] ** 2) ** 0.5), rot_36[2, 2]
        )
        theta_4 = np.arctan2(
            rot_36[1, 2] / np.sin(theta_5), rot_36[0, 2] / np.sin(theta_5)
        )
        theta_6 = np.arctan2(
            rot_36[2, 1] / np.sin(theta_5), -rot_36[2, 0] / np.sin(theta_5)
        )
    else:
        theta_5 = np.arccos(1)
        if np.arctan2(-rot_36[0, 1], rot_36[1, 1]) >= 0:
            theta_4 = np.arctan2(-rot_36[0, 1], rot_36[1, 1]) / 2
            theta_6 = np.arctan2(-rot_36[0, 1], rot_36[1, 1]) / 2
        else:
            theta_4 = np.arctan2(rot_36[1, 0], rot_36[0, 0]) / 2
            theta_6 = np.arctan2(rot_36[1, 0], rot_36[0, 0]) / 2

    return [np.degrees(theta_4), np.degrees(theta_5), np.degrees(theta_6)]


def get_rotation_matrix(vec2: np.ndarray, vec1: np.ndarray) -> np.ndarray:
    """get rotation matrix between two vectors using scipy"""

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))

    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()


def z_rot_matrix(x: float) -> np.ndarray:
    x = np.deg2rad(x)
    z_rot = [
        [np.cos(x), -np.sin(x), 0],
        [np.sin(x), np.cos(x), 0],
        [0, 0, 1],
    ]
    z_rot = np.array([[round(y, 10) for y in x] for x in z_rot])
    return z_rot


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    deg = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    return deg


def calc_rot_given_zx_vectors(z_vector: np.ndarray, x_vector: np.ndarray) -> list:
    # z_axis = np.array([0, 0, 1])
    # rot_z = get_rotation_matrix(z_vector, z_axis)

    # inv_rot_z = get_rotation_matrix(z_axis, z_vector)
    # inv_x_vector = inv_rot_z.dot(x_vector)

    # x_axis = np.array([1, 0, 0])
    # rot_x = get_rotation_matrix(x_vector, inv_x_vector)

    # rot_matrix = np.dot(rot_z, rot_x)

    z_axis = np.array([0, 0, 1])
    x_axis = np.array([1, 0, 0])
    rot_z = get_rotation_matrix(z_vector, z_axis)
    rotZ_xAxis = rot_z.dot(x_axis)

    angle_between_x = angle_between(rotZ_xAxis, x_vector)

    rot_z_forX = z_rot_matrix(angle_between_x)
    rot_matrix = rot_z.dot(rot_z_forX)

    check_xRot = [0, 0, 0] == [x - y for x, y in zip(x_vector, rot_matrix.dot(x_axis))]
    check_zRot = [0, 0, 0] == [x - y for x, y in zip(z_vector, rot_matrix.dot(z_axis))]

    if check_xRot == True and check_zRot == True:
        pass
    else:
        rot_z_forX = z_rot_matrix(-angle_between_x)
        rot_matrix = rot_z.dot(rot_z_forX)

    return rot_matrix


#%%
def avg_two_3d_points(pt1: list, pt2: list) -> list:
    avg_list = [(x + y) / 2 for x, y in zip(pt1, pt2)]
    return avg_list


def calc_z_vectors_from_joint_loc(stump_loc: list, mid_tip: list) -> np.ndarray:
    z_vec = np.array(mid_tip) - np.array(stump_loc)
    return z_vec


def calc_x_vectors_from_joint_loc(
    wrist: list,
    thb_tip: list,
    pky_tip: list,
) -> np.ndarray:
    thb_vec = np.array(thb_tip) - np.array(wrist)
    pky_vec = np.array(pky_tip) - np.array(wrist)
    x_vec = np.cross(pky_vec, thb_vec)
    return x_vec


def calc_claw_angle(wrist: list, mid_mcp: list, thb_tip: list) -> float:
    thb_vec = np.array(thb_tip) - np.array(wrist)
    mid_vec = np.array(mid_mcp) - np.array(wrist)
    joint_agl = angle_between(mid_vec, thb_vec)

    mult_angle = 1 * 1.75
    min_angle = 20
    joint_agl -= min_angle

    if joint_agl < min_angle:
        joint_agl = 0
    else:
        joint_agl = joint_agl * mult_angle

    return joint_agl


#%%
def bound_half_circle(agl: float) -> float:
    return_agl = agl

    if agl == -999:
        return_agl = -999

    if agl < -180:
        return_agl = agl + 360
    elif agl > 180:
        return_agl = agl - 360

    return return_agl


def bound_min_max(agl: float, bounds: dict) -> float:
    return_agl = agl

    if agl == -999:
        return_agl = -999

    if agl < bounds["min"]:
        return_agl = bounds["min"]
    elif agl > bounds["max"]:
        return_agl = bounds["max"]

    return_agl = round(return_agl, 1)

    return return_agl


def bound_angles(raw_angles: list, bounds_list: list) -> list:
    agls = raw_angles.copy()

    for n in range(len(raw_angles)):
        agls[n] = bound_min_max(agls[n], bounds_list[n])

    return agls


def calc_alt_wirst_angles(raw_angles: list, bounds_list: list) -> list:

    agls = [bound_half_circle(agl) for agl in raw_angles]
    agls = raw_angles
    agls_2 = agls.copy()

    if agls_2[5] < 0:
        agls_2[5] += 180
    else:
        agls_2[5] -= 180

    agls_3 = agls.copy()
    agls_4 = agls_2.copy()

    if agls_3[3] < 0:
        agls_3[3] += 180
        agls_3[4] = -agls_3[4]
    else:
        agls_3[3] -= 180
        agls_3[4] = -agls_3[4]

    if agls_4[3] < 0:
        agls_4[3] += 180
        agls_4[4] = -agls_4[4]
    else:
        agls_4[3] -= 180
        agls_4[4] = -agls_4[4]

    raw_angle_list = [agls, agls_2, agls_3, agls_4]

    adj_angle_list = []
    for raw_angle in raw_angle_list:
        adj_angle = bound_angles(raw_angle, bounds_list)
        adj_angle_list.append(adj_angle)

    return adj_angle_list


def calculate_angles_given_joint_loc(
    right_pts: dict,
    left_pts: dict,
    lens: list,
    bounds_list: list,
) -> list:

    l_wrist = left_pts["WRIST"]
    l_idx_mcp = left_pts["INDEX_FINGER_MCP"]
    l_mid_mcp = left_pts["MIDDLE_FINGER_MCP"]
    l_mid_tip = left_pts["MIDDLE_FINGER_TIP"]
    l_pky_mcp = left_pts["PINKY_MCP"]

    r_wrist = right_pts["WRIST"]
    r_thb_tip = right_pts["THUMB_TIP"]
    r_mid_tip = right_pts["MIDDLE_FINGER_TIP"]

    # get inital vectors given just locaitons
    z_vector = calc_z_vectors_from_joint_loc(l_mid_mcp, l_mid_tip)
    x_vector = calc_x_vectors_from_joint_loc(l_wrist, l_idx_mcp, l_pky_mcp)

    claw_agl = calc_claw_angle(r_wrist, r_mid_tip, r_thb_tip)
    arm_angles = calc_arm_angles(l_mid_mcp, lens)

    ## get wrist angle given z and x vectors
    # two options since can flip palm either way
    wirst_rot_06 = calc_rot_given_zx_vectors(z_vector, x_vector)
    wrist_angles = calc_wrist_angles(arm_angles + [0, 0, 0], lens, wirst_rot_06)

    ## CHANGED TO NEGATIVE SINCE STEPPERS MOVE IN CW DIR FOR POSTIVE ANGLE DEGREE
    wrist_angles = [-x for x in wrist_angles]

    raw_angles = arm_angles + wrist_angles + [claw_agl]
    raw_angles = [-999 if math.isnan(x) else x for x in raw_angles]

    adj_angles_list = calc_alt_wirst_angles(raw_angles, bounds_list)

    return adj_angles_list


def calc_end_rot_matrix_from_angles(adj_angles: list, lens: list) -> np.ndarray:
    # with all angles can just calc final angles and locations
    # mainly used for double checking
    DH_table = return_dh_table(adj_angles[:6], lens)
    homo_matrix_list = calc_homo_matrix(adj_angles[:6], DH_table)
    end_rot_matrix = calc_series_rotation(homo_matrix_list, 0, 6)

    return end_rot_matrix


# %%
lens = [8 + 13 + 4, 20, 13.5, 5.5, 0, 7]
bounds_list = [
    {"min": -90, "max": 200},
    {"min": 0, "max": 120},
    {"min": 0, "max": 135},
    {"min": -270, "max": 270},
    {"min": -120, "max": 120},
    {"min": -270, "max": 270},
    {"min": 0, "max": 135},
]

left_pts = {}
left_pts["WRIST"] = [10, 20, 20]
left_pts["INDEX_FINGER_MCP"] = [11, 20, 21]
left_pts["MIDDLE_FINGER_MCP"] = [10, 20, 21]
left_pts["MIDDLE_FINGER_TIP"] = [10, 20, 22]
left_pts["PINKY_MCP"] = [9, 20, 21]

right_pts = {}
right_pts["WRIST"] = [0, 0, 0]
right_pts["THUMB_TIP"] = [1, 0, 0]
right_pts["MIDDLE_FINGER_TIP"] = [0, 1, 0]


agls = calculate_angles_given_joint_loc(right_pts, left_pts, lens, bounds_list)
agls_0 = agls[0]
print("Given 3d Points: ")
for x in left_pts:
    print("\t", x, left_pts[x])
print("\n", "Return - Arm Joint Angles: ", agls_0[:3])
print("Return - Wrist Joint Angles: ", agls_0[3:6])

#%%


# agl_list = calculate_angles_given_joint_loc(
#     [10, 0, 15],
#     [29.540558916608976, 48.0685496768094, 28.078700303403046],
#     [33.300900141156646, 50.38910883622469, 27.41454766866296],
#     [32.631637794527386, 49.99106219329053, 27.52054288921509],
#     lengths,
#     bounds_list,
# )

# agl_list

# #%%
# lengths = [5, 8, 10, 2, 3, 1]
# a = lengths[0]
# b = lengths[1]
# c = lengths[2]
# d = lengths[3]
# e = lengths[4]
# f = lengths[5]

# bounds_list = [
#     {"min": -360, "max": 360},
#     {"min": -360, "max": 360},
#     {"min": -360, "max": 360},
#     {"min": -360, "max": 360},
#     {"min": -360, "max": 360},
#     {"min": -360, "max": 360},
#     {"min": -360, "max": 360},
# ]

# wrist_list = [
#     [b + c + d, 0, a],  # wrist: right down,
#     [b + c + d, 0, a],  # wrist: right down,
#     [b + c + d, 0, a],  # wrist: right down,
#     [b + c + d, 0, a],  # wrist: right down,
#     [b + c + d, 0, a],  # wrist: right down,
#     [b + c + d, 0, a],  # wrist: right down,
#     [b + c + d, 0, a],  # wrist: right down,
#     [b + c + d, 0, a],  # wrist: right down,
#     # [0, c + d, a + b],
#     # [0, c + d, a + b],
#     # [0, c + d, a + b],
#     # [0, c + d, a + b],
#     # [0, c + d, a + b],
#     # [0, c + d, a + b],
#     # [0, c + d, a + b],
#     # [0, c + d, a + b],
# ]

# pointer_pinky_thumb_list = [
#     [[0, -1, 1], [0, 1, 1]],  # up : horizontal
#     [[0, 0, 1], [1, 0, 0]],  # up : vertical
#     [[0, 0, -1], [0, 1, 0]],  # down : horizontal
#     [[0, 0, -1], [-1, 0, 0]],  # down : vertical
#     [[1, -1, 0], [1, 1, 0]],  # right : horizontal
#     [[1, 0, 1], [1, 0, -1]],  # right : vertical
#     [[0, 1, 0], [0, 0, -1]],  # front : horizontal
#     [[0, 1, 0], [-1, 0, 0]],  # front : vertical
# ]

# wrist_cord_add = [
#     [0, 0, e + f],
#     [0, 0, e + f],
#     [0, 0, -(e + f)],
#     [0, 0, -(e + f)],
#     [(e + f), 0, 0],
#     [(e + f), 0, 0],
#     [0, (e + f), 0],
#     [0, (e + f), 0],
# ]

# end_cords_xyz = [np.array(x) + np.array(y) for x, y in zip(wrist_list, wrist_cord_add)]

# combo_cords = []
# for n in range(len(wrist_list)):
#     wrist_cord = np.array(wrist_list[n])
#     adj_pp_list = [np.add(x, wrist_cord) for x in pointer_pinky_thumb_list[n]]
#     combo_cords.append([wrist_cord] + (adj_pp_list))
# print(end_cords_xyz)

# #%%
# adj_angle_list = []
# for x, exp_cord in zip(combo_cords, end_cords_xyz):
#     adj_angle_list = calculate_angles_given_joint_loc(
#         x[0], x[1], x[0], x[2], lengths, bounds_list
#     )
#     inverse_neg = [1, 1, 1, -1, -1, -1, 1]
#     for adj_agl in adj_angle_list:
#         adj_agl = [x*y for x, y in zip(inverse_neg,adj_agl)]
#         end_rot_matrix = calc_end_rot_matrix_from_angles(adj_agl, lengths)
#         mtx = np.array(end_rot_matrix)
#         end_cord = np.array([mtx[0, 3], mtx[1, 3], mtx[2, 3]])
#         print((end_cord == exp_cord).all(), end_cord, exp_cord)

#     print(np.array(adj_angle_list))

# #%%
# lengths = [8, 20, 13.5, 5.5, 0, 7]
# lengths = [5, 8, 10, 2, 3, 1]
# test_angles = [0, 0, 0, 0, 90, 0]
# # test_angles = [62.1, 52.4, 72.8, 0, 0, 0]
# DH_table = return_dh_table(test_angles, lengths)
# homo_matrix_list = calc_homo_matrix(test_angles, DH_table)
# end_rot_matrix = calc_series_rotation(homo_matrix_list, 0, 6)
# print(end_rot_matrix)

# #%%
# points = {
#     "WRIST": [11.72807113, 24.63063475, 22.40443979],
#     "THUMB_TIP": [13.99622764, 21.17608839, 30.81874532],
#     "INDEX_FINGER_TIP": [13.17943863, 21.89371136, 35.53726186],
#     "PINKY_TIP": [12.33898948, 25.56899693, 33.98904994],
# }
# mul_agls = [
#     [64.5, 64.4, 75.5, 11.2, 108.7, 129.2, 13.0],
#     [64.5, 64.4, 75.5, 11.2, 108.7, -50.8, 13.0],
#     [64.5, 64.4, 75.5, -168.8, -108.7, 129.2, 13.0],
#     [64.5, 64.4, 75.5, -168.8, -108.7, -50.8, 13.0],
# ]
# agls = [64.5, 64.4, 75.5, 11.2, 108.7, -50.8, 13.0]

# points = {
#     "WRIST": [10.84282841, 21.42719212, 24.79908365],
#     "THUMB_TIP": [13.32789476, 19.88487063, 15.53860106],
#     "INDEX_FINGER_TIP": [15.05010878, 24.06773044, 12.26389772],
#     "PINKY_TIP": [11.5183303, 29.09501156, 16.91070188],
# }
# mul_agls = [
#     [63.2, 75.0, 82.6, 167.8, 64.1, 89.7, 20.7],
#     [63.2, 75.0, 82.6, 167.8, 64.1, -90.3, 20.7],
#     [63.2, 75.0, 82.6, -12.2, -64.1, 89.7, 20.7],
#     [63.2, 75.0, 82.6, -12.2, -64.1, -90.3, 20.7],
# ]
# agls = [63.2, 75.0, 82.6, 167.8, 64.1, -90.3, 20.7]

# points = {
#     "WRIST": [15.61401201, 21.59682316, 26.22006716],
#     "THUMB_TIP": [23.36391067, 25.78619491, 29.15633429],
#     "INDEX_FINGER_TIP": [23.95958934, 27.58383403, 31.34801129],
#     "PINKY_TIP": [18.29192804, 27.98011351, 31.38054358],
# }
# mul_agls = [
#     [63.7, 27.3, 135, 94.6, 120, 116.9, 10.5],
#     [63.7, 27.3, 135, 94.6, 120, -63.1, 10.5],
#     [63.7, 27.3, 135, -85.4, -120, 116.9, 10.5],
#     [63.7, 27.3, 135, -85.4, -120, -63.1, 10.5],
# ]
# agls = [63.7, 27.3, 135, -85.4, -120, -63.1, 10.5]


#%%
# adj_angle_list = calculate_angles_given_joint_loc(
#     points["WRIST"],
#     points["PINKY_TIP"],
#     points["INDEX_FINGER_TIP"],
#     points["THUMB_TIP"],
#     lengths,
#     bounds_list,
# )

# end_rot_matrix = calculate_end_rot_matrix_from_angles(
#     adj_angle_list[0], [8, 20, 13.5, 5.5, 0, 7]
# )
# print(end_rot_matrix)
# calc_arm_angles(points["WRIST"], lengths)

# %%
# lengths = [8, 20, 13.5, 5.5, 0, 7]
# mul_agls = [
#     [63.7, 27.3, 135, 94.6, 120, 116.9, 10.5],
#     [63.7, 27.3, 135, 94.6, 120, -63.1, 10.5],
#     [63.7, 27.3, 135, -85.4, -120, 116.9, 10.5],
#     [63.7, 27.3, 135, -85.4, -120, -63.1, 10.5],
# ]

# for n in mul_agls:
#     DH_table = return_dh_table(n[:6], lengths)
#     homo_matrix_list = calc_homo_matrix(test_angles, DH_table)
#     end_rot_matrix = calc_series_rotation(homo_matrix_list, 0, 6)
#     print(end_rot_matrix)


# %%
# wrist_loc, pointer_loc, pinky_loc, thumb_loc = combo_cords[4]
# # get inital vectors given just locaitons
# z_vector, x_vector, x_vector_2 = calc_x_z_vectors_from_joint_loc(
#     wrist_loc, pinky_loc, pointer_loc
# )

# closed_agl = 10
# max_agl = 100
# claw_agl = calc_claw_angle(wrist_loc, pointer_loc, thumb_loc, closed_agl, max_agl)

# arm_angles = calc_arm_angles(wrist_loc, lengths)

# ## get wrist angle given z and x vectors
# # two options since can flip palm either way
# wrist_angles_list = []
# for x_vec in [x_vector, x_vector_2]:
#     wirst_rot_06 = calc_rot_given_zx_vectors(z_vector, x_vec)
#     wrist_angles = calc_wrist_angles(arm_angles + [0, 0, 0], lengths, wirst_rot_06)
#     wrist_angles_list.append(wrist_angles)


# # wirst_rot_06 = calc_rot_given_zx_vectors(z_vector, x_vector)

# z_axis = np.array([0, 0, 1])
# x_axis = np.array([1, 0, 0])
# rot_z = get_rotation_matrix(z_vector, z_axis)
# rotZ_xAxis = rot_z.dot(x_axis)


# angle_between_x = angle_between(rotZ_xAxis, x_vector)

# rot_z_forX = z_rot_matrix(angle_between_x)
# wirst_rot_06 = rot_z.dot(rot_z_forX)

# check_xRot = [0, 0, 0] == [x - y for x, y in zip(x_vector, wirst_rot_06.dot(x_axis))]
# check_zRot = [0, 0, 0] == [x - y for x, y in zip(z_vector, wirst_rot_06.dot(z_axis))]

# if check_xRot == True and check_zRot == True:
#     pass
# else:
#     rot_z_forX = z_rot_matrix(-angle_between_x)
#     wirst_rot_06 = rot_z.dot(rot_z_forX)


# #%%
# DH_table = return_dh_table(arm_angles + [0, 0, 0], lengths)

# homo_matrix_list = calc_homo_matrix(arm_angles + [0, 0, 0], DH_table)

# rot_03 = calc_series_rotation(homo_matrix_list, 0, 3)[:3, :3]
# inv_rot_03 = rot_03.transpose()
# rot_36 = inv_rot_03.dot(wirst_rot_06)

# print(z_vector, x_vector)
# print(wirst_rot_06)
# print(rot_03)
# print(inv_rot_03)
# print(rot_36)

# z_axis = np.array([0, 0, 1])
# x_axis = np.array([1, 0, 0])
# print(
#     wirst_rot_06.dot(x_axis),
#     x_vector,
#     [0, 0, 0] == [x - y for x, y in zip(x_vector, wirst_rot_06.dot(x_axis))],
# )
# print(
#     wirst_rot_06.dot(z_axis),
#     z_vector,
#     [0, 0, 0] == [x - y for x, y in zip(z_vector, wirst_rot_06.dot(z_axis))],
# )

# if sum(1 for i in wrist_angles_list[0] if i >= 0) > sum(
#     1 for i in wrist_angles_list[1] if i >= 0
# ):
#     wrist_angles = wrist_angles_list[0]
# else:
#     wrist_angles = wrist_angles_list[1]

# print(wrist_angles)
# #%%
# # combine raw angles and adjust for negative values
# # and out of bound angles

# angles = arm_angles + wrist_angles
# angles = [0 if math.isnan(x) else x for x in angles]

# raw_angles = [round(x) for x in angles]
# # adj_angles = bound_angles(angles, bounds_list)

# # with all angles can just calculate final angles and locations
# # mainly used for double checking
# DH_table = return_dh_table(raw_angles, lengths)
# homo_matrix_list = calc_homo_matrix(raw_angles, DH_table)
# end_rot_matrix = calc_series_rotation(homo_matrix_list, 0, 6)
# print(end_rot_matrix)

# # %%
# def get_rotation_matrix(vec2, vec1):
#     """get rotation matrix between two vectors using scipy"""
#     vec1 = vec1 / np.linalg.norm(vec1)
#     vec2 = vec2 / np.linalg.norm(vec2)

#     vec1 = np.reshape(vec1, (1, -1))
#     vec2 = np.reshape(vec2, (1, -1))

#     r = R.align_vectors(vec2, vec1)
#     return r[0].as_matrix()


# z_axis = np.array([0, 0, 1])
# rot_z = get_rotation_matrix(z_vector, z_axis)

# inv_rot_z = get_rotation_matrix(z_axis, z_vector)
# inv_x_vector = inv_rot_z.dot(x_vector)

# x_axis = np.array([1, 0, 0])
# rot_x = get_rotation_matrix(x_vector, inv_x_vector)

# rot_matrix = np.dot(rot_z, rot_x)

# [print(x) for x in [rot_z, inv_rot_z, rot_x, rot_matrix]]
# print(rot_matrix.dot(x_axis), x_vector)
# print(rot_matrix.dot(z_axis), z_vector)


# DH_table = return_dh_table(arm_angles + [0, 0, 0], lengths)

# homo_matrix_list = calc_homo_matrix(arm_angles + [0, 0, 0], DH_table)

# rot_03 = calc_series_rotation(homo_matrix_list, 0, 3)[:3, :3]
# inv_rot_03 = np.linalg.inv(rot_03)
# rot_36 = np.dot(inv_rot_03, rot_matrix)


#%%
# # ----
# # TEST ARM ANGLES
# # ----
# END_CORD = [0, 8, 6]
# lens = [1, 5, 7, 1, 1, 2]
# calc_arm_angles(END_CORD, lens)

# #%%
# # ----
# # GIVEN ANGLES - GET END ROTATION MATRIX
# # use calculated rotation matrix to check if correct angles output wrist
# # ----
# lens = [1, 5, 7, 1, 1, 2]
# arm_angles = [30, 40, 70]
# angles = arm_angles + [20, 84, -49.99999999871018]
# DH_table = return_dh_table(angles, lens)

# homo_matrix_list = calc_homo_matrix(angles, DH_table)

# rot_06_test = calc_series_rotation(homo_matrix_list, 0, 6)[:3, :3]
# test_x = np.dot(
#     calc_series_rotation(homo_matrix_list, 0, 6)[:3, :3], np.array([1, 0, 0])
# )
# test_z = np.dot(
#     calc_series_rotation(homo_matrix_list, 0, 6)[:3, :3], np.array([0, 0, 1])
# )
# print(test_x, test_z)
# calc_series_rotation(homo_matrix_list, 0, 6)

# #%%
# calc_wrist_angles(arm_angles + [0, 0, 0], lens, rot_06_test)

# #%%
# # ----
# # GIVEN FECTORS FOR WRIST - GET ROTATION MATRIX AND ANGLES NEEDED TO ACHIEVE
# # Should have same rotation matrix as above is same angles
# # ----
# z_vector = np.array([0, 1, 1])
# x_vector = np.array([0, 1, -1])

# arm_angles = [90, 90, 90, 0, 0, 0]
# rot_test = calc_rot_given_zx_vectors(z_vector, x_vector)
# print(calc_wrist_angles(arm_angles, lens, rot_test))
# print(rot_test)

# # ----
# # GIVEN LOCATIONS CALCULATE Z/X vectors (pointer and palm)
# # ----
# wrist_loc = [1, 1, 1]
# pinky_loc = [0, 3, 5]
# pointer_loc = [2, 4, 7]

# calc_x_z_vectors_from_joint_loc(wrist_loc, pinky_loc, pointer_loc)
