#%%
import numpy as np
import math


def law_of_cos(e1, e2, opp):
    if e1 > 0 or e2 > 0:
        opp_angle = np.arccos((e1 ** 2 + e2 ** 2 - opp ** 2) / (2 * e1 * e2))
        return opp_angle
    else:
        return 0


def calc_arm_angles(END_CORD, lens):
    X, Y, Z = END_CORD[0], END_CORD[1], END_CORD[2]

    if X == 0 and Y == 0:
        return [0, 0, 0]

    else:
        theta_1 = np.arcsin(Y / (X ** 2 + Y ** 2) ** 0.5)

        z_adj = Z - lens[0]
        d = sum([X ** 2, Y ** 2, z_adj ** 2]) ** 0.5

        theta_21 = np.arctan(z_adj / (X ** 2 + Y ** 2) ** 0.5)
        theta_22 = law_of_cos(lens[1], d, lens[2] + lens[3])
        theta_2 = theta_21 + theta_22

        theta_3 = np.radians(180) - law_of_cos(lens[1], lens[2] + lens[3], d)

        arm_angles = [np.degrees(theta_1), np.degrees(theta_2), np.degrees(theta_3)]

    return arm_angles


def calc_series_rotation(homo_matrix_list, first_value, last_value):
    first_matrix = True
    homo_matrix_slice = homo_matrix_list[first_value:last_value]

    for homo_matrix in homo_matrix_slice:
        if first_matrix == True:
            overall_rot_matrix = homo_matrix
            first_matrix = False
        else:
            overall_rot_matrix = np.dot(overall_rot_matrix, homo_matrix)

    return overall_rot_matrix


def calc_homo_matrix(angles, DH_table):

    num_joints = len(angles)
    homo_matrix_list = [0] * num_joints

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


def return_dh_table(angles, lens):
    DH_table = [
        [angles[0] + 0, 90, 0, lens[0]],
        [angles[1] + 0, 180, lens[1], 0],
        [angles[2] - 90, -90, 0, 0],
        [angles[3] + 0, 90, 0, lens[2] + lens[3]],
        [angles[4] + 90, 90, 0, 0],
        [angles[5] + 0, 0, 0, lens[4] + lens[5]],
    ]
    return DH_table


def calc_wirst_rot_matrix(wrist_angles):
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


def calc_wrist_angles(arm_angles, lens, rot_06):
    DH_table = return_dh_table(arm_angles, lens)

    homo_matrix_list = calc_homo_matrix(arm_angles, DH_table)

    rot_03 = calc_series_rotation(homo_matrix_list, 0, 3)[:3, :3]
    inv_rot_03 = np.linalg.inv(rot_03)
    rot_36 = np.dot(inv_rot_03, rot_06)

    theta_5 = np.arcsin(rot_36[2, 2])
    if round(theta_5, 10) != round(np.pi / 2, 10):
        if rot_36[0, 2] == 0:
            theta_4 = np.pi / 2
        else:
            theta_4 = np.arctan(rot_36[1, 2] / rot_36[0, 2])
        if rot_36[2, 0] == 0:
            theta_6 = np.pi / 2
        else:
            theta_6 = np.arctan(-rot_36[2, 1] / rot_36[2, 0])
    else:
        half_angle = np.arctan(rot_36[1, 0] / rot_36[0, 0]) / 2
        theta_6 = half_angle
        theta_4 = half_angle

    return [np.degrees(theta_4), np.degrees(theta_5), np.degrees(theta_6)]


def align_vectors(vector_to_align, align_to_vector):
    align_to_vector = align_to_vector / np.linalg.norm(align_to_vector)
    vector_to_align = vector_to_align / np.linalg.norm(vector_to_align)
    v = np.cross(vector_to_align, align_to_vector)
    c = np.dot(vector_to_align, align_to_vector)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R


def calc_rot_given_zx_vectors(z_vector, x_vector):
    z_axis = np.array([0, 0, 1])
    rot_z = align_vectors(z_axis, z_vector)

    inv_rot_z = align_vectors(z_vector, z_axis)
    inv_x_vector = inv_rot_z.dot(x_vector)

    x_axis = np.array([1, 0, 0])
    rot_x = align_vectors(x_axis, inv_x_vector)

    rot_matrix = np.dot(rot_z, rot_x)
    return rot_matrix


#%%


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    deg = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    if math.isnan(deg):
        return 999
    else:
        return deg


def calc_x_z_vectors_from_joint_loc(wrist_loc, pinky_loc, pointer_loc):
    pinky_vec = np.array(pinky_loc) - np.array(wrist_loc)
    pointer_vec = np.array(pointer_loc) - np.array(wrist_loc)
    palm_vec = np.cross(pinky_vec, pointer_vec)
    palm_vec_2 = np.cross(pointer_vec, pinky_vec)
    return pointer_vec, palm_vec, palm_vec_2


def calc_claw_angle(wrist_loc, pointer_loc, thumb_loc, closed_agl, max_agl):
    thumb_vec = np.array(thumb_loc) - np.array(wrist_loc)
    pointer_vec = np.array(pointer_loc) - np.array(wrist_loc)
    joint_agl = int(round(angle_between(thumb_vec, pointer_vec)))

    if joint_agl < closed_agl:
        return 0
    elif joint_agl == 999:
        return 255
    elif joint_agl > max_agl:
        return max_agl
    else:
        return joint_agl


# ----
# TEST ARM ANGLES
# ----
# END_CORD = [0, 8, 6]
# lens = [1, 5, 7, 1, 1, 2]
# calc_arm_angles(END_CORD, lens)

# ----
# GIVEN ANGLES - GET END ROTATION MATRIX
# use calculated rotation matrix to check if correct angles output wrist
# ----
# #%%
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
# calc_wrist_angles(arm_angles + [0,0,0], lens, rot_06_test)

# #%%
# ----
# GIVEN FECTORS FOR WRIST - GET ROTATION MATRIX AND ANGLES NEEDED TO ACHIEVE
# Should have same rotation matrix as above is same angles
# ----
# z_vector = np.array([0, 1, 1])
# x_vector = np.array([0, 1, -1])

# arm_angles = [90, 90, 90, 0, 0, 0]
# rot_test = calc_rot_given_zx_vectors(z_vector, x_vector)
# print(calc_wrist_angles(arm_angles, lens, rot_test))
# print(rot_test)

# ----
# GIVEN LOCATIONS CALCULATE Z/X vectors (pointer and palm)
# ----
# wrist_loc = [1, 1, 1]
# pinky_loc = [0, 3, 5]
# pointer_loc = [2, 4, 7]

# calc_x_z_vectors_from_joint_loc(wrist_loc, pinky_loc, pointer_loc)
# def bound_angles(raw_angles_list, bounds_list):
#     adj_angles_list = raw_angles_list.copy()
#     adjusted_angles = 1
#     while adjusted_angles > 0:
#         adjusted_angles = 0
#         for n in range(len(raw_angles_list)):
#             if adj_angles_list[n] < bounds_list[n]["min"]:
#                 adj_angle = bounds_list[n]["min"]
#                 adjusted_angles += 1
#             elif adj_angles_list[n] > bounds_list[n]["max"]:
#                 adj_angle = bounds_list[n]["max"]
#                 adjusted_angles += 1
#             else:
#                 adj_angle = adj_angles_list[n]
#             adj_angles_list[n] = int(round(adj_angle))

#     return adj_angles_list


def min_max_bound(agl, bounds):
    if agl < bounds["min"]:
        return bounds["min"]
    elif agl > bounds["max"]:
        return bounds["max"]
    else:
        return agl


def bound_angles(raw_angles_list, bounds_list):
    agls = raw_angles_list.copy()
    agls = [int(round(x)) for x in agls]

    for n in [0, 1, 2, 4]:
        min_max_bound(agls[n], bounds_list[n])

    if agls[3] < 0:
        agls[3] = 180 + agls[3]
        agls[4] = 180 - agls[4]
    if agls[5] < 0:
        agls[5] = 180 + agls[5]

    for n in [3, 4, 5]:
        agls[n] = min_max_bound(agls[n], bounds_list[n])

    return agls


#%%
def calculate_angles_given_joint_loc(
    wrist_loc, pinky_loc, pointer_loc, thumb_loc, lens, bounds_list
):
    # get inital vectors given just locaitons
    z_vector, x_vector, x_vector_2 = calc_x_z_vectors_from_joint_loc(
        wrist_loc, pinky_loc, pointer_loc
    )

    closed_agl = 10
    max_agl = 100
    claw_agl = calc_claw_angle(wrist_loc, pointer_loc, thumb_loc, closed_agl, max_agl)

    arm_angles = calc_arm_angles(wrist_loc, lens)

    ## get wrist angle given z and x vectors
    # two options since can flip palm either way
    wrist_angles_list = []
    for x_vec in [x_vector, x_vector_2]:
        wirst_rot_06 = calc_rot_given_zx_vectors(z_vector, x_vec)
        wrist_angles = calc_wrist_angles(arm_angles + [0, 0, 0], lens, wirst_rot_06)
        wrist_angles_list.append(wrist_angles)

    if sum(1 for i in wrist_angles_list[0] if i < 0) > sum(
        1 for i in wrist_angles_list[1] if i < 0
    ):
        wrist_angles = wrist_angles_list[1]
    else:
        wrist_angles = wrist_angles_list[0]

    # combine raw angles and adjust for negative values
    # and out of bound angles

    angles = arm_angles + wrist_angles
    angles = [0 if math.isnan(x) else x for x in angles]

    raw_angles = [round(x) for x in angles]
    adj_angles = bound_angles(angles, bounds_list)

    # with all angles can just calculate final angles and locations
    # mainly used for double checking
    DH_table = return_dh_table(adj_angles, lens)
    homo_matrix_list = calc_homo_matrix(adj_angles, DH_table)
    end_rot_matrix = calc_series_rotation(homo_matrix_list, 0, 6)

    return raw_angles, adj_angles, end_rot_matrix, claw_agl


# bounds_list = [
#     {"min": 0, "max": 180},
#     {"min": 0, "max": 135},
#     {"min": 0, "max": 135},
#     {"min": 0, "max": 180},
#     {"min": 0, "max": 180},
#     {"min": 0, "max": 180},
# ]

# wrist_loc = [3, 8, 6]
# pinky_loc = [2, 10, 8]
# pointer_loc = [4, 11, 9]
# thumb_loc = [5, 9, 6]
# lens = [1, 5, 7, 1, 1, 2]

# raw_angles, end_angles, end_rot_matrix, claw_agl = calculate_angles_given_joint_loc(
#     wrist_loc, pinky_loc, pointer_loc, thumb_loc, lens, bounds_list
# )
# print(raw_angles)
# print(end_angles)
# print(end_rot_matrix)
# print(claw_agl)
# %%